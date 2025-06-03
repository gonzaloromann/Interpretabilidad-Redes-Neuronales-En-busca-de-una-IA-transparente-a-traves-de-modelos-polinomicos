data <- read.csv("C:/Users/gonza/Downloads/Medicaldataset.csv")
p=ncol(data)-1
data$Result=as.factor(data$Result)
data$Result=ifelse(data$Result=="positive",1,0)



año=seq(1997,2023)
altas_n=c(486531,508496,540433,551885,563301,580812,593340,598985,600527,600949,610449,616260,
          611127,615355,606498,610390,618633,623921,628563,603521,611691,612066,614302,524016,
          582446,584830,587918)
          
altas_u=c(290136,318695,345650,351893,385893,402817,399183,407954,411720,415348,424882,428122,
          426241,431718,432849,441243,447305,451003,458705,433023,447614,450358,453253,391909,
          439493,431057,435538)

top_n=c(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,0,1,0,1,1,0)
top_u=c(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,1,1,0)

# Cargar librerías necesarias
library(ggplot2)
library(dplyr)
library(scales)  # Para formatear los porcentajes en el eje Y

# Supongamos que ya tienes estos vectores definidos:
# altas_n, altas_u: vectores numéricos
# top_n, top_u: vectores binarios (0 y 1)
# año: vector con los años

# Crear un data frame con toda la información
datos <- data.frame(
  año = año,
  altas_n = altas_n,
  altas_u = altas_u,
  top_n = top_n,
  top_u = top_u
)

# Crear una columna categórica para cada punto que indique qué color le corresponde
datos_n <- datos %>%
  transmute(año, valor = altas_n,
            categoria = ifelse(top_n == 1, "1ª causa de altas", "2ª causa de altas"),
            grupo = "Altas Hospitalarias")

datos_u <- datos %>%
  transmute(año, valor = altas_u,
            categoria = ifelse(top_u == 1, "1ª causa de altas urgentes", "2ª causa de altas urgentes"),
            grupo = "Altas Hospitalarias Urgentes")

# Unir ambos conjuntos
datos_tidy <- bind_rows(datos_n, datos_u)

# Crear el gráfico
ggplot() +
  # Líneas de evolución
  geom_line(data = datos, aes(x = año, y = altas_n), color = "grey40", size = 1) +
  geom_line(data = datos, aes(x = año, y = altas_u), color = "grey70", size = 1) +
  
  # Puntos coloreados
  geom_point(data = datos_tidy,
             aes(x = año, y = valor, color = categoria, shape = grupo),
             size = 3) +
  
  # Colores y formas
  scale_color_manual(
    name = "Categoría",
    values = c(
      "1ª causa de altas" = "#FFD700",
      "2ª causa de altas" = "#C0C0C0",
      "1ª causa de altas urgentes" = "#1f78b4",
      "2ª causa de altas urgentes" = "#999999"
    )
  ) +
  scale_shape_manual(
    name = "Serie",
    values = c("Altas Hospitalarias" = 16, "Altas Hospitalarias Urgentes" = 17)  # 16: círculo, 17: triángulo
  ) +
  
  labs(
    title = "Evolución de altas y altas urgentes por año",
    x = "Año",
    y = "Valores"
  ) +
  theme_minimal()

# Calcular el porcentaje de altas_u sobre altas_n
datos <- datos %>%
  mutate(porcentaje = ifelse(altas_n == 0, NA, altas_u / altas_n))

ggplot(datos, aes(x = año, y = porcentaje)) +
  geom_line(color = "#1f78b4", size = 1) +          # Línea azul
  geom_point(color = "#1f78b4", size = 3) +         # Puntos azules
  scale_y_continuous(labels = percent_format(accuracy = 1)) +  # Formato de porcentaje en el eje Y
  labs(
    title = "Evolución del porcentaje de altas urgentes sobre el total",
    x = "Año",
    y = "Porcentaje"
  ) +
  theme_minimal()
