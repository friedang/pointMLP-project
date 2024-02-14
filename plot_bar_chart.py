import matplotlib.pyplot as plt

# Data
x_labels = ['PTv3','PointNet',"PointMLP", "Self  \n attention \n modification", "Multihead Attention \n with color features"]
y_data = [
    [0.87, 0.80],
    [0.78, 0.43],
    [0.67, 0.52],
    [0.70, 0.56],
    [0.70, 0.57],
]

# Plotting
fig, ax = plt.subplots()
bar_width = 0.2
index = range(len(x_labels))

for i in range(len(y_data[0])):
    plt.bar([x + i * bar_width for x in index], [data[i] for data in y_data], bar_width, label=f'Test {["Accuracy", "Class mIOU"][i]}')

# plt.xlabel('Models')
plt.ylabel('Scores')
plt.title('Test Scores by Model and Metric')
plt.xticks([i + bar_width for i in index], x_labels, ha="center")
plt.legend()

plt.tight_layout()

plt.savefig('test_scores_bar_chart_comparison.png')

plt.show()
