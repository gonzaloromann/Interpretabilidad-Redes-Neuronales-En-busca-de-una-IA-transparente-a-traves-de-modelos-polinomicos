library(torch)
library(luz)
library(nn2poly)

batches=c(32,50,64,128)
epocas=c(300,500,700,900)

inicio=0
resultado=c()
  
for (batch in batches){
   for (epoca in epocas){
     for(i in 1:10){
    
       data <- read.csv("Medicaldataset.csv")
       p=ncol(data)-1
       data$Result=as.factor(data$Result)
       data$Result=ifelse(data$Result=="positive",1,0)
       
       # Data scaling to [-1,1]
       maxs <- apply(data[,-(p+1)], 2, max)
       mins <- apply(data[-(p+1)], 2, min)
       data[,-(p+1)] <- as.data.frame(scale(data[,-(p+1)], center = mins + (maxs - mins) / 2, scale = (maxs - mins) / 2))
       
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
         train = luz::as_dataloader(only_train_list, batch_size = batch, shuffle = TRUE),
         valid = luz::as_dataloader(val_list, batch_size = batch)
       )
       
       luz_nn <- function() {
         torch::torch_manual_seed(100473601)
         
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
           loss = torch::nn_bce_with_logits_loss(),
           optimizer = torch::optim_adam,
           metrics = luz::luz_metric_binary_accuracy() 
         ) %>%
         add_constraints("l1_norm") %>%
         fit(torch_data$train, epochs = epoca, valid_data = torch_data$valid)
       
       precisión=fitted_con$records$metrics$valid[[epoca]]$acc
       
       if(precisión>inicio){
         resultado=c(precisión,batch,epoca)
         inicio=precisión
         print('Nueva mejor precisión')
       }
       cat('Mejor precisión:', inicio, ' con parámetros: Batch:',batch,'y Épocas:',epoca)
     }
   }
 }
  
  

fitted_con %>% plot()






