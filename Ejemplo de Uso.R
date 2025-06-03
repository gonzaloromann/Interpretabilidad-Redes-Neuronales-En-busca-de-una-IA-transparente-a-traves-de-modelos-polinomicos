set.seed(100473601)
polynomial <- list()

#7x1+3x2^2-4x1x3
polynomial$labels <- list(c(1), c(2,2),c(1,3))
polynomial$values <- c(3,4,-7)

# Define number of variables and sample size
p <- 3
n_sample <- 500

# Predictor variables
X <- matrix(0,n_sample,p)
for (i in 1:p){
  X[,i] <- rnorm(n = n_sample,0,1)
}

# Response variable + small error term
Y <- nn2poly:::eval_poly(poly = polynomial, newdata = X) +
  stats::rnorm(n_sample, 0, 0.1)

# Store all as a data frame
data <- as.data.frame(cbind(X, Y))

# Data scaling to [-1,1]
maxs <- apply(data, 2, max)
mins <- apply(data, 2, min)
data <- as.data.frame(scale(data, center = mins + (maxs - mins) / 2, scale = (maxs - mins) / 2))

# Divide in train (0.75) and test (0.25)
index <- sample(1:nrow(data), round(0.75 * nrow(data)))
train <- data[index, ]
test <- data[-index, ]

train_x <- as.matrix(train[,-(p+1)])
train_y <- as.matrix(train[,(p+1)])

test_x <- as.matrix(test[,-(p+1)])
test_y <- as.matrix(test[,(p+1)])

# Divide in only train and validation
all_indices   <- 1:nrow(train_x)
only_train_indices <- sample(all_indices, size = round(nrow(train_x)) * 0.8)
val_indices   <- setdiff(all_indices, only_train_indices)

# Create lists with x and y values to feed luz::as_dataloader()
only_train_x <- as.matrix(train_x[only_train_indices,])
only_train_y <- as.matrix(train_y[only_train_indices,])
val_x <- as.matrix(train_x[val_indices,])
val_y <- as.matrix(train_y[val_indices,])

only_train_list <- list(x = only_train_x, y = only_train_y)
val_list <- list(x = val_x, y = val_y)

torch_data <- list(
  train = luz::as_dataloader(only_train_list, batch_size = 50, shuffle = TRUE),
  valid = luz::as_dataloader(val_list, batch_size = 50)
)


luz_nn <- function() {
  torch::torch_manual_seed(42)
  
  luz_model_sequential(
    torch::nn_linear(p,100),
    torch::nn_tanh(),
    torch::nn_linear(100,100),
    torch::nn_tanh(),
    torch::nn_linear(100,100),
    torch::nn_tanh(),
    torch::nn_linear(100,1)
  )
}

nn_con <- luz_nn()

fitted_con <- nn_con %>%
  luz::setup(
    loss = torch::nn_mse_loss(),
    optimizer = torch::optim_adam,
  ) %>%
  add_constraints("l1_norm") %>%
  fit(torch_data$train, epochs = 700, valid_data = torch_data$valid)

fitted_con %>% plot()

# Obtain the predicted values with the NN to compare them
prediction_NN_con <- as.array(predict(fitted_con, test_x))

# Diagonal plot implemented in the package to quickly visualize and compare predictions
nn2poly:::plot_diagonal(x_axis =  prediction_NN_con, y_axis =  test_y, xlab = "Constrained NN prediction", ylab = "Original Y")

# Polynomial for nn_con
final_poly_con <- nn2poly(object = fitted_con,
                          max_order = 3)

prediction_poly_con <- predict(object = final_poly_con, newdata = test_x)


nn2poly:::plot_diagonal(x_axis =  prediction_NN_con, y_axis =  prediction_poly_con, xlab = "NN prediction", ylab = "Polynomial prediction") + ggplot2::ggtitle("Polynomial for nn_con")

plot(final_poly_con, n = 8)

