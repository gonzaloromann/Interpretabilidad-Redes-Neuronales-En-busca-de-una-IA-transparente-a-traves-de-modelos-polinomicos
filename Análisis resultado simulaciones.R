resultados=readRDS("C:/Users/gonza/OneDrive/Escritorio/Universidad/4to curso/TFG/5.Simulaciones/resultados.rds")

library(dplyr)

resultados %>%
  group_by(orden) %>%
  summarise(
    RMSE_medio = mean(RMSE),
    precision_media = mean(precisión_signo),
    tiempo_total_medio = mean(tiempo_total),
    tiempo_polinomio_medio = mean(tiempo_polinomio)
  )
library(ggplot2)

# Error vs número de variables
ggplot(resultados, aes(x = as.factor(variables), y = RMSE)) +
  geom_boxplot(fill='lightcoral',color='darkred') +
  labs(title = "Distribución del RMSE según número de variables",x='Variables')



ggplot(resultados, aes(x = as.factor(orden), y = RMSE)) +
  geom_boxplot(fill='lightcoral',color='darkred') +
  labs(title = "Distribución del RMSE por orden", x = "Orden", y = "RMSE")




ggplot(resultados, aes(x = as.factor(orden), y = precisión_signo)) +
  geom_violin(fill = "lightgreen", color = "darkgreen") +
  geom_boxplot(width = 0.1, fill = "white") +  # opcional: agrega boxplot dentro del violín
  labs(title = "Gráfico de violín de la precisión del signo por orden del polinomio",
       x = "Orden del polinomio", y = "Precisión Signo")


ggplot(subset(resultados, orden == 2), aes(x = as.factor(variables), y = tiempo_polinomio)) +
  geom_boxplot(fill = "orange", alpha = 0.2) +
  stat_summary(fun = mean, geom = "line", aes(group = 1), color = "orange", size = 1) +
  stat_summary(fun = mean, geom = "point", color = "orange", size = 3) +
  labs(title = "Distribución y promedio del tiempo del polinomio por número de variables(orden 2)",
       x = "Número de variables", y = "Tiempo del polinomio") +
  theme_minimal()

ggplot(subset(resultados, orden == 3), aes(x = as.factor(variables), y = tiempo_polinomio)) +
  # Boxplots por cada número de variables
  geom_boxplot(fill = "skyblue", alpha = 0.2) +
  
  # Línea de promedios y puntos (por grupo de variables)
  stat_summary(fun = mean, geom = "line", aes(group = 1), color = "skyblue", size = 1) +
  stat_summary(fun = mean, geom = "point", color = "skyblue", size = 3) +
  
  labs(title = "Distribución y promedio del tiempo del polinomio por número de variables (orden 3)",
       x = "Número de variables", y = "Tiempo del polinomio") +
  theme_minimal()

ggplot(resultados, aes(x = as.factor(variables), y = tiempo_polinomio)) +
  # Boxplots superpuestos por orden (sin separación horizontal)
  geom_boxplot(aes(fill = as.factor(orden)), alpha = 0.3, position = position_identity()) +
  
  # Línea de medias por orden
  stat_summary(aes(group = as.factor(orden), color = as.factor(orden)),
               fun = mean, geom = "line", size = 1) +
  
  # Puntos de medias por orden
  stat_summary(aes(color = as.factor(orden)),
               fun = mean, geom = "point", size = 3) +
  
  labs(title = "Distribución y promedio del tiempo del polinomio por número de variables",
       x = "Número de variables", y = "Tiempo del polinomio",
       fill = "Orden", color = "Orden") +
  theme_minimal()

# 5. Precisión del signo según número de monomios, coloreando por orden
ggplot(resultados, aes(x = num_mon, y = precisión_signo, color = as.factor(orden))) +
  geom_point() +
  geom_hline(yintercept=0.5,linetype='dashed',color='red')+
  labs(title = "Precisión del signo según el número de monomios",
       x = "Número de monomios", y = "Precisión del signo", color = "Orden")
# 6. Precisión del signo según número de variables, coloreando por orden
ggplot(resultados, aes(x = as.factor(variables), y = precisión_signo, fill = as.factor(orden))) +
  geom_boxplot(position = position_dodge(width = 0.75)) +
  labs(title = "Precisión del signo según el número de variables",
       x = "Número de variables", y = "Precisión del signo", fill = "Orden") +
  theme_minimal()

t.test(resultados[resultados$orden==2,]$precisión_signo,
       resultados[resultados$orden==3,]$precisión_signo,
       alternative = "two.sided",var.equal = FALSE)
