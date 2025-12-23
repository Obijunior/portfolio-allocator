import matplotlib.pyplot as plt


def plot_cumulative_returns(results):
    plt.figure(figsize=(10, 6))

    for name, returns in results.items():
        cumulative = (1 + returns).cumprod()
        plt.plot(cumulative, label=name)

    plt.legend()
    plt.title("Cumulative Returns")
    plt.xlabel("Date")
    plt.ylabel("Growth of $1")
    plt.grid(True)
    plt.show()
