import pandas as pd

MANIFEST_PATH = "data/processed/manifest.csv"

df = pd.read_csv(MANIFEST_PATH)

# Number of participants
participants = sorted(df["participant_id"].unique())

print("Number of participants:", len(participants))
print("Number of recordings:", len(df))

# Recordings contributed by each participant
participant_counts = (
    df.groupby("participant_id")
      .size()
      .sort_values(ascending=False)
)

print("\nRecordings per participant:")
print(participant_counts)

# Class distribution
print("\nOverall class distribution:")
print(df["sign"].value_counts().sort_index())

# Participant x class matrix
participant_class = pd.crosstab(
    df["participant_id"],
    df["sign"]
)

print("\nParticipant-class matrix:")
print(participant_class)

# Check which participants contribute to each class
print("\nNumber of participants per class:")
print((participant_class > 0).sum(axis=0))
