import pandas as pd

df = pd.DataFrame({
    "subject_id": [1001, 1001, 1001, 1001, 1002, 1002],
    "timestamp": ["2019-01-05 12:00", "2019-01-05 13:13", "2019-01-05 14:07", "2019-01-05 14:17", "2019-01-05 11:17", "2019-01-05 11:58"],
    "event_type": [0, 2, 1, 4, 0, 5],
    "icd_code": ["", "0059", "J11", "", "", ""],
    "icd_version": ["", "9", "10", "", "", ""]
})

class TokenConverter:
    def convert_row_to_token_seq(self, row): 
        """
        Convert one row into a token.
        """
        event = row["event_type"]

        # 0 - admission
        if(event == 0):
            return self.adm_to_token()
        # 1 - diagnose
        elif(event == 1):
            icd_code = row["icd_code"]
            icd_version = row["icd_version"]
            is_icd_10 = True if icd_version == "10" else False
            return self.diag_to_token(icd_code, is_icd_10)
        # 2 - procedure
        elif (event == 2):
            icd_code = row["icd_code"]
            icd_version = row["icd_version"]
            is_icd_10 = True if icd_version == "10" else False
            return self.proc_to_token(icd_code, is_icd_10)
        # 3 - medication
        elif (event == 3):
            drug_cd = row["formulary_drug_cd"]
        # 4 - redamission
        elif (event == 4):
            return self.readm_to_token()
        # 5 - death
        elif (event == 5):
            return self.death_to_token()

    def adm_to_token(self):
        return "[ADM]"
    
    def readm_to_token(self):
        return "[READM]"
    
    def death_to_token(self):
        return "[DEATH]"
        
    def diag_to_token(self, icd_code: str, is_icd_10: bool):
        return f"[DIAG_{icd_code}]" if is_icd_10 else f"[DIAG9_{icd_code}]"
    
    def proc_to_token(self, icd_code: str, is_icd_10: bool):
        return f"[PROC_{icd_code}]" if is_icd_10 else f"[PROC9_{icd_code}]"
    
    def med_to_token(self, drug_cd: str):
        return f"[MED_{drug_cd}]"

converter = TokenConverter()
patient = 0
seq = ""
for i in range(0, len(df)):
    if (patient != df.iloc[i]["subject_id"]):
        patient = df.iloc[i]["subject_id"]
        seq = ""
        print("----------------------------------")
    seq += (converter.convert_row_to_token_seq(df.iloc[i]))
    if(df.iloc[i]["event_type"] != 4 and df.iloc[i]["event_type"] != 5):
        seq += (" -> ")
    else:
        print(seq)
