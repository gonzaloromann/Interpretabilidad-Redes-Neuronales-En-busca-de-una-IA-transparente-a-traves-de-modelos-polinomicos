library(nn2poly)
library(torch)
library(luz)
library(tictoc)
set.seed(42)

resultados=data.frame(orden=integer(),
                      variables = integer(),
                      num_mon = integer(),
                      RMSE=numeric(),
                      precisión_signo=numeric(),
                      tiempo_total= numeric(),
                      tiempo_polinomio=numeric(),
                      stringsAsFactors = FALSE)

nume_mon=c(20,40,60,80,100)
variables=c(10,15,20)
orden=3
rep=1

for(o in orden){
  for(v in variables){
    for (m in nume_mon){
      for(i in 1:rep){
        tic()
        
        
          num_mon=round(runif(1,m-5,m))
          inicio=v-3
          vector_valores=c()
          label=list()
          for(j in 1:num_mon){
            vector_valores=c(vector_valores,round(runif(1,-10,10),3))
            label[[j]]=sample(inicio:v, o)
        } 
        
        
        
        #print(label)
        #etiquetas_unicas=sort(unique(unlist(label)))
        #print(etiquetas_unicas)
        #label=lapply(label, function(x) match(x, etiquetas_unicas))
        #print(label)
        
        polynomial <- list()
        polynomial$labels <- label
        polynomial$values <- vector_valores
        
        p=max(unlist(label))
        n_sample <- 500
        
        # Predictor variables
        X <- matrix(0,n_sample,p)
        for (k in 1:p){
          X[,k] <- rnorm(n = n_sample,0,1)
        }
        
        # Response variable + small error term
        Y <- nn2poly:::eval_poly(poly = polynomial, newdata = X) +
          stats::rnorm(n_sample, 0, 0.1)
        
        # Store all as a data frame
        data <- as.data.frame(cbind(X, Y))
        head(data)
        
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
          fit(torch_data$train, epochs = 2, valid_data = torch_data$valid)
        
        #fitted_con %>% plot()
        
        
        # Obtain the predicted values with the NN to compare them
        prediction_NN_con <- as.array(predict(fitted_con, test_x))
        
        
        
        # Diagonal plot implemented in the package to quickly visualize and compare predictions
        # nn2poly:::plot_diagonal(x_axis =  prediction_NN_con, y_axis =  test_y, xlab = "Constrained NN prediction", ylab = "Original Y")
        
        
        cat("numero monomios :", num_mon)
        cat("numero variables :",p)
        # Polynomial for nn_con
        if(o==3){
          tic()
          final_poly_con <- nn2poly(object = fitted_con,
                                    max_order = 3)
          tiempo_polinomio=toc(quiet=TRUE)
        } else{
          tic()
          final_poly_con <- nn2poly(object = fitted_con,
                                    max_order = 2)
          tiempo_polinomio=toc(quiet=TRUE)
        }
        
        
        
        
        prediction_poly_con <- predict(object = final_poly_con, newdata = test_x)
        
        
        
        rmse=sqrt(mean((prediction_NN_con - prediction_poly_con)^2))
        
        coeficientes_finales=c()
        vec=c()
        
        for(b in 1:length(polynomial$labels)){
          for(c in 1:length(final_poly_con$labels)){
            if(identical(polynomial$labels[[b]],final_poly_con$labels[[c]])){
              vec=c(polynomial$values[b],final_poly_con$values[c])
              coeficientes_finales=rbind(coeficientes_finales,vec)
            }
          }
        }
        
        
        
        acc_signo=mean(sign(coeficientes_finales[,1])==sign(coeficientes_finales[,2]))
        
        tiempo_total=toc(quiet=TRUE)
        
        resultados[nrow(resultados) + 1, ] = c(o,p,num_mon, rmse, acc_signo,
                                               tiempo_total$toc - tiempo_total$tic,
                                               tiempo_polinomio$toc - tiempo_polinomio$tic)
       
        
        cat("Final Iteración",i)
        
      }
    }
  }
}
