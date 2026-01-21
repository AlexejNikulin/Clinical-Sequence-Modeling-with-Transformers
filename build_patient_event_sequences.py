import pandas as pd
from typing import List, Tuple

from datetime import datetime

import networkx as nx

import matplotlib.pyplot as plt

import pydot
from IPython.display import Image

from pathlib import Path

from vocabulary import Vocabulary
from tqdm import tqdm

class EventSequencer():
    def __init__(self):
        
        self.SOURCE_TO_STAGE = {
            "admissions": "Admission",
            "labevents": "Labevent",
            "diagnoses_icd": "Diagnosis",
            "emar": "Medication",
            "discharge": "Discharge",
            "death": "Discharge",
            "patients": "Patient Data",
        }

    
        self.TIME_GAP_BINS = [
        (0.0,       1/24,     0), #≤1h
        (1/24,      3/24,     1), #1–3h
        (3/24,      12/24,    2), #3–12h
        (12/24,     1.0,      3), #12–24h
        (1.0,       3.0,      4), #1–3d
        (3.0,       7.0,      5), #3–7d
        (7.0,       28.0,     6), #1–4w
        (28.0,      90.0,     7), #1–3mo
        (90.0,      365.0,    8), #3–12mo
        (365.0,     1095.0,   9), #1-3y
        (1095.0,    36500.0, 10), #3-100y
        (36500.0,   float("inf"), 11) #<100y mainly to catch the differnce between Dem events and other ones
        ]

    def build_patient_event_sequences(
        self,
        df: pd.DataFrame,
        vocab
    ) -> List[List[str]]:
        """
        Convert an event table into time-ordered event sequences per patient.
    
        Expected columns:
            - subject_id
            - timestamp
            - event_type
            - event_value
            - source
    
        Returns:
            List[List[str]]: one ordered list of event strings per subject_id
        """
        sequences = []
    
        df = df.copy()
    
        for subject_id, group in df.groupby("subject_id"):
            patient_sequences = [[], []]

            for _, row in group.iterrows():
    
                event = vocab.row_to_token(row)
    
                if row["event_type"] == "DEM":
                    patient_sequences[0].append(event)
                else:
                    patient_sequences[1].append(event)
    
            sequences.append(patient_sequences)
    
        return sequences

    def categorize_time_gap(self, gap_days: float) -> str:
        for lower, upper, label in self.TIME_GAP_BINS:
            if lower <= gap_days <= upper:
                return label
        return "unknown"

    
    def add_time_tokens_to_data(self, df: pd.DataFrame):
        rows = []

        previous_subject = None
        previous_timestamp = None
        previous_event_type = None
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        idx = 0

        for _, row in df.iterrows():
            current_subject = row["subject_id"]
            current_timestamp = row["timestamp"]
            current_event_type = row["event_type"]

            if(row["event_type"] != "TIME"):
                if(previous_event_type != "DEM"):
                    if previous_subject != current_subject:
                        gap_category = "start"
                        previous_timestamp = None
                    else:
                        gap_days = (current_timestamp.to_pydatetime() - previous_timestamp.to_pydatetime()).total_seconds() / 86400
                        gap_category = self.categorize_time_gap(gap_days)

                    if gap_category != 0:
                        rows.append({
                            "subject_id": current_subject,
                            "timestamp": current_timestamp.to_pydatetime() - pd.Timedelta(seconds=1),
                            "event_type": "TIME",
                            "event_value": gap_category,
                        })

                rows.append(row.to_dict())

            previous_subject = current_subject
            previous_timestamp = current_timestamp
            previous_event_type = current_event_type

        out = (
            pd.DataFrame(rows)
            .sort_values(["subject_id", "timestamp"])
            .reset_index(drop=True)
        )

        out.to_csv(
            Path("../out/merge_and_sort/combined.csv"), 
            mode='w', 
            header=True, 
            index=False
        )

    def build_stage_sequence_with_counts(self, df: pd.DataFrame, subject_id: int):
        """
        Returns a list of (stage, first_timestamp, count) tuples,
        collapsing consecutive identical stages **only if they occur on the same day**.
        """
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df[df["subject_id"] == subject_id]
        df = df.sort_values("timestamp")

        sequence = []

        last_stage = None
        count = 0
        first_ts = None
        last_date = None

        for _, row in df.iterrows():
            stage = self.SOURCE_TO_STAGE.get(row["source"])
            if stage is None:
                continue

            ts = row["timestamp"]
            ts_date = ts.date()  # only the day

            # collapse only if stage is same AND date is same
            if stage == last_stage and ts_date == last_date:
                count += 1
            else:
                if last_stage is not None:
                    sequence.append((last_stage, first_ts, count))

                last_stage = stage
                count = 1
                first_ts = ts
                last_date = ts_date

        # append the last stage
        if last_stage is not None:
            sequence.append((last_stage, first_ts, count))

        return sequence


    def build_transition_graph(self, stage_sequence):
        G = nx.DiGraph()

        for i, (stage, ts, count) in enumerate(stage_sequence):
            # label with stage + count + timestamp on 2 lines
            label = stage
            if count > 1:
                label += f" ({count})"
            
            # split timestamp into two lines
            label += f"\n{ts.strftime('%Y-%m-%d')}\n{ts.strftime('%H:%M:%S')}"

            G.add_node(i, label=label)

            if i > 0:
                G.add_edge(i - 1, i)

        return G


    def draw_stage_graph(self, G, output_file="stage_graph.png"):
        """
        Draws a stage transition graph using pydot/Graphviz.
        Exports the graph to a PNG file with visible arrowheads.
        """

        dot = pydot.Dot(graph_type="digraph", rankdir="LR", nodesep="1.0", ranksep="1.0")

        # add nodes
        for i, data in G.nodes(data=True):
            label = data.get("label", str(i))
            node = pydot.Node(
                str(i),
                label=label,
                shape="box",
                style="filled",
                fillcolor="#1f2937",
                fontcolor="white",
                fontsize=11
            )
            dot.add_node(node)

        # add edges with arrowheads
        for source, target in G.edges():
            edge = pydot.Edge(
                str(source),
                str(target),
                color="#111827",
                arrowhead="vee",  # solid arrowhead
                penwidth=2
            )
            dot.add_edge(edge)

        # Write to PNG file
        dot.write_png(output_file)
        print(f"Graph saved to {output_file}")


    def visualize_sequence(self, df, subject_id):
        """
        Combines the above functions to visualize the event trajectory of a single patient in a png
        """

        
        df_sub = df[df["subject_id"] == subject_id]

        print(df_sub["source"].value_counts())

        stage_sequence = self.build_stage_sequence_with_counts(df, subject_id)
        G = self.build_transition_graph(stage_sequence)
        print(stage_sequence)
        
        G.graph['graph'] = {
            'rankdir': 'LR',    # Left-to-right layout
            'nodesep': '1.0',   # horizontal spacing between nodes
            'ranksep': '1.0'    # vertical spacing between ranks (not critical here)
        }
        
        self.draw_stage_graph(G)


    # Example:
    # COMBINED_CSV = Path("../out/merge_and_sort/combined.csv")
    # df = pd.read_csv(COMBINED_CSV)

    # subject_id = 10000032
    # visualize_sequence(df, subject_id)




