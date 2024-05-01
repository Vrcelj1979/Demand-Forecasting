# Demand-Forecasting

Example of a simple Python program to forecast product demand using a regression model, specifically a linear regression model. This can be useful for inventory or production planning. For this example, we will use the scikit-learn library, which is easy to use and offers a variety of machine learning algorithms.

Before we begin, make sure you have the following libraries installed using pip:

numpy (for working with numerical data)
pandas (for data processing)
scikit-learn (for building a regression model)
matplotlib (to visualize the results)

In this example, we created a simple linear regression model that tries to predict sales (Sales) based on price (Price) and advertising budget (Ad_Budget). The data is presented in the form of a dictionary, and then we convert it to a pandas DataFrame for further processing.

Using train_test_split, we separated the data into training and test sets. Then we created a LinearRegression model and trained it on the training data (X_train, y_train). With this model, we then predicted the values for the test data (X_test) and compared the results with the actual values (y_test).

Finally, we visualized the results with a graph, where the actual values (y_test) are on the x-axis, and the predicted values (y_pred) are on the y-axis. The graph helps us assess how well the model predicts demand.

Graph display:
1.Save the graph as an image (main.py): Instead of using plt.show(), you can save the graph as an image. When you run the program, the graph will be saved as an image prediction_graph.png in the same folder where your program is located. You can then open this image on your computer or in a terminal if it supports image display.
2. Use a different method to display the graph (main2.py) If you still want to display the graph in the terminal, you can try using another method to display the graph, such as using io.BytesIO and the PIL (Pillow) library to convert and display the image in the terminal. An example is in the document main.py, after the command: python main.py the graph will be drawn automatically.
3.In chrome using flask (main3.py)


