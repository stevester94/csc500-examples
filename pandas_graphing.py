#! /usr/bin/env python3
data = """
{
  "source": [
    true,
    true,
    true,
    true,
    true,
    true,
    true,
    false,
    false,
    false,
    false,
    false,
    false,
    false,
    false,
    false,
    false,
    false,
    false,
    false
  ],
  "accuracy": [
    0.09426987060998152,
    0.1059988351776354,
    0.22732696897374702,
    0.33557046979865773,
    0.3695106649937265,
    0.3337313432835821,
    0.3290282902829028,
    0.3337349397590361,
    0.3247549019607843,
    0.32510288065843623,
    0.29484029484029484,
    0.08869987849331713,
    0.31828703703703703,
    0.3297166968053044,
    0.08830694275274056,
    0.0998162890385793,
    0.11592434411226357,
    0.17144563918757466,
    0.3197747183979975,
    0.3546583850931677
  ],
  "domain": [
    -18,
    -12,
    -6,
    0,
    6,
    12,
    18,
    2,
    4,
    8,
    10,
    -20,
    14,
    16,
    -16,
    -14,
    -10,
    -8,
    -4,
    -2
  ]
}
"""


import json
import pandas as pds
import matplotlib.pyplot as plt

data = json.loads(data)

fig, ax = plt.subplots()
fig.set_size_inches(30, 15)

df = pds.DataFrame.from_dict(data)
df = df.sort_values("domain")


# groupby no good, index is lost
# df = df.set_index("domain", drop=True)
# df.groupby("source").plot(kind="bar", ax=ax, xticks=df.index, use_index=True)


# Explodes
# df.groupby("source").plot(kind="bar", x="domain", y="accuracy", ax=ax, xticks="domain", use_index=False)


# x index is just fucked
# df[df["source"] == True].plot(kind="bar", x="domain", y="accuracy", color="blue", ax=ax)
# df[df["source"] == False].plot(kind="bar",x="domain", y="accuracy", color="red", ax=ax, xticks=df["domain"]) # The second plot indices are overwriting the first

df = df.pivot(index="domain", columns="source", values="accuracy")
df.plot(kind="bar")
# df.plot(kind="bar", x="domain", y="accuracy", color="blue", ax=ax)

# df[df["source"] == True].plot(kind="bar", x="domain", y="accuracy", color="blue", ax=ax)
# df[df["source"] == False].plot(kind="bar",x="domain", y="accuracy", color="red", ax=ax)


plt.show()