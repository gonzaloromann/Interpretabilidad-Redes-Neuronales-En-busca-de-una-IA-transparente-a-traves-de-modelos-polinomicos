library(torch)
library(luz)
library(nn2poly)
library(caret)
library(ggplot2)

# --- 1. Data Loading and Preprocessing ---
data <- read.csv("C:/Users/gonza/Downloads/Medicaldataset.csv")
p <- ncol(data) - 1 # Number of predictors

# Convert Result to numeric 0 and 1
data$Result <- as.factor(data$Result)
summary(data$Result)
data$Result <- ifelse(data$Result == "positive", 1, 0)

# Data scaling to [-1,1] for predictors
maxs <- apply(data[, -(p + 1)], 2, max)
mins <- apply(data[, -(p + 1)], 2, min)
data_scaled_x <- as.data.frame(scale(data[, -(p + 1)], center = mins + (maxs - mins) / 2, scale = (maxs - mins) / 2))
processed_data <- cbind(data_scaled_x, Result = data$Result)

# Divide in train (0.75) and test (0.25)
set.seed(100473601) # for reproducibility
index <- sample(1:nrow(processed_data), round(0.75 * nrow(processed_data)))
train_df <- processed_data[index, ]
test_df <- processed_data[-index, ]

train_x_matrix <- as.matrix(train_df[, -(p + 1)])
train_y_matrix <- as.matrix(train_df[, (p + 1)]) 

test_x_matrix <- as.matrix(test_df[, -(p + 1)])
test_y_vector <- test_df[, (p + 1)] # Keep as vector for confusion matrix later

# Divide train into only_train and validation
all_indices <- 1:nrow(train_x_matrix)
only_train_indices <- sample(all_indices, size = round(nrow(train_x_matrix) * 0.8))
val_indices <- setdiff(all_indices, only_train_indices)

only_train_x <- train_x_matrix[only_train_indices, ]
only_train_y <- train_y_matrix[only_train_indices, , drop = FALSE] # Keep as matrix
val_x <- train_x_matrix[val_indices, ]
val_y <- train_y_matrix[val_indices, , drop = FALSE] # Keep as matrix

# Torch data loader
torch_data <- list(
  train = luz::as_dataloader(list(x = only_train_x, y = only_train_y), batch_size = 64, shuffle = TRUE),
  valid = luz::as_dataloader(list(x = val_x, y = val_y), batch_size = 64)
)

# --- 2. Neural Network Definition ---
# For binary classification with nn_bce_with_logits_loss,
# the output layer should have 1 neuron.
luz_nn_binary <- function() {
  torch::torch_manual_seed(100473601)
  luz_model_sequential(
    torch::nn_linear(p, 100),
    torch::nn_tanh(),
    torch::nn_linear(100, 100),
    torch::nn_tanh(),
    torch::nn_linear(100, 100),
    torch::nn_tanh(),
    torch::nn_linear(100, 1) # Output 1 logit for the positive class
  )
}

nn_model <- luz_nn_binary()

# --- 3. Training the Neural Network ---
fitted_nn_model <- nn_model %>%
  luz::setup(
    loss = torch::nn_bce_with_logits_loss(), # Expects raw logits and float targets (0.0 or 1.0)
    optimizer = torch::optim_adam,
    metrics = luz::luz_metric_binary_accuracy_with_logits() # Use _with_logits version
  ) %>%
  add_constraints("l1_norm") %>%
  fit(torch_data$train, epochs = 900, valid_data = torch_data$valid) # Reduced epochs for quick test

# Plot training history
plot(fitted_nn_model)

# --- 5. Predictions and Evaluation ---
# Get raw logits from the neural network
nn_logits_test <- predict(fitted_nn_model, test_x_matrix)

# Convert logits to probabilities
nn_probs_test <- torch::torch_sigmoid(nn_logits_test)

# Convert probabilities to predicted classes
nn_predicted_classes_tensor <- (nn_probs_test > 0.5)$to(dtype = torch_long())
nn_predicted_classes_r <- as.array(nn_predicted_classes_tensor)

message("Confusion Matrix for Neural Network (with constraints):")
cm_nn <- caret::confusionMatrix(
  data = factor(nn_predicted_classes_r, levels = c(0, 1)),
  reference = factor(test_y_vector, levels = c(0, 1)),
  positive = "1"
)
print(cm_nn)

fourfoldplot(cm_nn$table,
             color = c("#e74c3c", "#2ecc71"),
             conf.level = 0,
             margin = 1,
             main = "Matriz de Confusión")


# --- 6. Using nn2poly ---
final_poly <- nn2poly(
  object = fitted_nn_model,
  max_order = 3 # Vignettes often use max_order = 3, can experiment
)

# Get polynomial predictions (logits)
poly_logits_test_r <- predict(object = final_poly, newdata = test_x_matrix)


# --- Diagonal Plot: Comparing NN Logits to Polynomial Logits ---
nn_logits_test_r <- as.array(nn_logits_test)

# Ensure the diagonal plot function adds the y=x line
print(nn2poly:::plot_diagonal(
  x_axis = nn_logits_test_r,
  y_axis = poly_logits_test_r,
  xlab = "NN Logits",
  ylab = "Polynomial Logits"
) + ggplot2::ggtitle("NN Logits vs. Polynomial Logits") 
)


# You can also get polynomial-based class predictions:
poly_probs_test <- 1 / (1 + exp(-poly_logits_test_r)) # Manual sigmoid for R numeric logits
poly_predicted_classes_r <- ifelse(poly_probs_test > 0.5, 1, 0)

message("Confusion Matrix for Polynomial Approximation:")
cm_poly <- caret::confusionMatrix(
  data = factor(poly_predicted_classes_r, levels = c(0, 1)),
  reference = factor(test_y_vector, levels = c(0, 1)),
  positive = "1"
)
print(cm_poly)

fourfoldplot(cm_poly$table,
             color = c("#e74c3c", "#2ecc71"),
             conf.level = 0,
             margin = 1,
             main = "Matriz de Confusión")

# Plot other nn2poly visualizations
plot(final_poly, n = 8) # Plot most important coefficients

nn_probs_test <- as.numeric(nn_probs_test)

df.1 <- data.frame(
  Polynomial = poly_probs_test,
  NeuralNet = nn_probs_test
)

# Crear gráfico
ggplot(df.1, aes(x = Polynomial, y = NeuralNet)) +
  annotate("rect", xmin = 0, xmax = 0.5, ymin = 0.5, ymax = 1,
           alpha = 0.2, fill = "steelblue")+
  geom_point(color = "black", alpha = 0.6) +         # puntos dispersos
  geom_abline(slope = 1, intercept = 0, color = "red") +  # línea identidad
  labs(
    title = "Comparación de Probabilidades Predichas",
    x = "Probabilidades - Modelo Polinómico",
    y = "Probabilidades - Red Neuronal"
  ) +
  theme_minimal() +                                      # tema más limpio
  theme(
    plot.title = element_text(hjust = 0.5, size = 14, face = "bold")
  )


# Create a combined data frame for ggplot
df.2 <- data.frame(
  Probability = c(nn_probs_test,poly_probs_test),
  Model = factor(c(rep("Red Neuronal", length(nn_probs_test)),
                   rep("Polinomio", length(poly_probs_test))))
)

# Create the boxplot
ggplot(df.2, aes(x = Model, y = Probability, fill = Model)) +
  geom_boxplot() +
  scale_fill_manual(values = c( "Red Neuronal" = "lightgreen","Polinomio" = "lightblue")) +
  labs(title = "Distribución Probabilidades",
       y = "Probabilidad Predicha",
       x = NULL) +
  theme_minimal(base_size = 14)
