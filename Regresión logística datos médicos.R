# Cargar librerías necesarias
library(caret)
library(dplyr)
library(ggplot2)

data <- read.csv("C:/Users/gonza/Downloads/Medicaldataset.csv")
p <- ncol(data) - 1 # Number of predictors
# Asegúrate de que Result es factor para clasificación
data$Result <- as.factor(data$Result)

data$Result <- ifelse(data$Result == "positive", 1, 0)
data=processed_data
set.seed(100473601)  # Para reproducibilidad

# --- 1. División 75% train / 25% test ---
trainIndex <- sample(1:nrow(data), round(0.75 * nrow(data)))
train_data <- data[trainIndex, ]
test_data <- data[-trainIndex, ]

# --- 2. División interna del 75%: 80% / 20% ---
trainIndex_internal <- sample(1:nrow(train_data),round(0.75*nrow(train_data)))
train_internal <- train_data[trainIndex_internal, ]
val_internal <- train_data[-trainIndex_internal, ]

set.seed(100473601)
# --- 3. Ajustar modelo de regresión logística ---
modelo_log <- glm(Result ~ ., data = train_internal, family = binomial)

summary(modelo_log)
# --- 4. Predecir sobre test final ---
prob_test <- predict(modelo_log, newdata = test_data, type = "response")
pred_test <- ifelse(prob_test > 0.5, "1", "0") |> as.factor()

# --- 5. Evaluar modelo ---
conf_mat <- caret::confusionMatrix(
  data = factor(pred_test, levels = c(0, 1)),
  reference = factor(test_data$Result, levels = c(0, 1)),
  positive = "1"
)


print(conf_mat)
fourfoldplot(conf_mat$table,
             color = c("#e74c3c", "#2ecc71"),
             conf.level = 0,
             margin = 1,
             main = "Matriz de Confusión")

# Extraer métricas
precision <- conf_mat$byClass["Precision"]
recall <- conf_mat$byClass["Recall"]
f1 <- conf_mat$byClass["F1"]
accuracy <- conf_mat$overall["Accuracy"]

# Mostrar métricas
cat("\n--- Métricas ---\n")
cat("Precisión :", round(precision, 3), "\n")
cat("Recall    :", round(recall, 3), "\n")
cat("F1 Score  :", round(f1, 3), "\n")
cat("Accuracy  :", round(accuracy, 3), "\n")



library(ggplot2)
library(broom)

# Extraer coeficientes y p-valores
coef_df <- tidy(modelo_log)

# Eliminar intercepto
coef_df <- coef_df[coef_df$term != "(Intercept)", ]

# Asignar color según significancia
coef_df$color <- ifelse(coef_df$p.value < 0.05, "significativo", "no_significativo")

# Definir colores pastel
colores <- c("significativo" = "#6BAED6", "no_significativo" = "#FDAE6B")  # azul pastel y naranja pastel

# Graficar
ggplot(coef_df, aes(x = estimate, y = reorder(term, estimate), fill = color)) +
  geom_col(width = 0.6, show.legend = FALSE) +
  scale_fill_manual(values = colores) +
  geom_vline(xintercept = 0, linetype = "dashed", color = "gray40") +
  labs(
    title = "Coeficientes del Modelo de Regresión Logística",
    x = "Coeficientes",
    y = "Variables"
  ) +
  theme_minimal()

coef_df <- coef_df[abs(coef_df$estimate) < 2, ]

# Asignar color según significancia
coef_df$color <- ifelse(coef_df$p.value < 0.05, "significativo", "no_significativo")

# Definir colores pastel
colores <- c("significativo" = "#6BAED6", "no_significativo" = "#FDAE6B")  # azul pastel y naranja pastel

# Graficar
ggplot(coef_df, aes(x = estimate, y = reorder(term, estimate), fill = color)) +
  geom_col(width = 0.6, show.legend = FALSE) +
  scale_fill_manual(values = colores) +
  geom_vline(xintercept = 0, linetype = "dashed", color = "gray40") +
  labs(
    x = "Coeficientes",
    y = "Variables"
  ) +
  theme_minimal()
