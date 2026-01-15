import pandas as pd
from typing import List, Tuple

from datetime import datetime

import networkx as nx

import matplotlib.pyplot as plt

import pydot
from IPython.display import Image

from pathlib import Path

from vocabulary import Vocabulary

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


    def build_patient_event_sequences(
        self,
        df: pd.DataFrame,
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

        self.MAX_TIME_GAP = 0.1
        
        VOCAB_PATH = Path("../out/vocab/vocabulary.json")
        vocab = Vocabulary.load(VOCAB_PATH)

        sequences = []

        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        for subject_id, group in df.groupby("subject_id"):
            patient_sequences = [[], []]

            previous_timestamp = None

            for _, row in group.iterrows():
                current_timestamp = row["timestamp"].to_pydatetime()
                event = vocab.row_to_token(row)

                if previous_timestamp is not None:
                    gap_days = (current_timestamp - previous_timestamp).total_seconds() / 86400
                else:
                    gap_days = 0.0

                # Make sure that last event is neither too close nore a DEM event
                # if gap_days >= self.MAX_TIME_GAP and gap_days <= 36500:
                #     # append Time Token
                #     continue

                if row["event_type"] == "DEM":
                    patient_sequences[0].append(event)
                else:
                    patient_sequences[1].append(event)

                previous_timestamp = current_timestamp

            sequences.append(patient_sequences)

        return sequences


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



