import pandas as pd

with open("slurm-665066.out") as f:
    slurm = f.readlines()

df = pd.DataFrame(slurm)
df = df.rename(columns={0:'raw'})

df['progress'] = df.raw.str.extract("Progress:\s*(\d+.\d)%")
df['loss'] = df.raw.str.extract("avg\.loss:\s*(\d+\.\d+)")
df['words_sec_thread'] = df.raw.str.extract("thread:\s*(\d+)\s")
df['learning_rate'] = df.raw.str.extract("lr:\s*(\d+\.\d+)\s")

df = df.dropna().reset_index(drop=True)

df.progress = df.progress.astype(float)
df.loss = df.loss.astype(float)
df.words_sec_thread = df.words_sec_thread.astype(int)
df.learning_rate = df.learning_rate.astype(float)

df = df.drop(columns="raw")