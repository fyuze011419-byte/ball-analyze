##################### Load Packages (For Analysis) ####################
## Libraries Path
.libPaths()
.libPaths("C:/Program Files/R/R-4.5.1/library")
.libPaths()

## Install Packages (Run Once)
install.packages(c("rpart", "rpart.plot", "randomForest", "xgboost",
                   "caTools", "ggplot2", "GGally", "gridExtra"))

## Load Packages
library(rpart)        # For Decision Tree (CART)
library(rpart.plot)   # For Tree Visualization
library(randomForest) # For Random Forest
library(xgboost)      # For XGBoost
library(caTools)      # For Data Splitting
library(ggplot2)      # For Data Visualization
library(GGally)       # For Scatter Plot Matrix
library(gridExtra)    # For Plot Arrangement


#############################################################################
## Baseball Salary Data (For Regression)
## Response : Logsalary = log(Salary)
## Models   : Multiple Linear Regression, CART, Random Forest, XGBoost
#############################################################################

## Load Data (ball.txt 放在「下載」資料夾)
## Sys.getenv("USERPROFILE") = C:/Users/你的使用者名稱
file.ball <- file.path(Sys.getenv("USERPROFILE"), "Downloads", "ball.txt")
file.ball
file.exists(file.ball)   # 應該要是 TRUE
## 若為 FALSE，請直接改成完整路徑，例如：
## file.ball <- "C:/Users/你的使用者名稱/Downloads/ball.txt"

Baseball <- read.table(
  file.ball,
  header = TRUE,
  sep = "",
  quote = "\"",
  dec = ".",
  stringsAsFactors = TRUE   # 文字欄位轉為 factor
)

## Descriptive Statistics
dim(Baseball)
summary(Baseball)
str(Baseball)
head(Baseball, n = 10)

## Missing Values
colSums(is.na(Baseball))


########################################################################
## Data Visualization
########################################################################

## Salary vs. Logsalary (取 log 後較接近常態分配)
h1 <- ggplot(Baseball, aes(x = Salary)) +
  geom_histogram(bins = 30, fill = "steelblue", color = "white") +
  labs(title = "Salary (thousands)") +
  theme_bw()

h2 <- ggplot(Baseball, aes(x = Logsalary)) +
  geom_histogram(bins = 30, fill = "darkorange", color = "white") +
  labs(title = "Log Salary") +
  theme_bw()

windows()
grid.arrange(h1, h2, ncol = 2)

## Boxplots of Logsalary by Categorical Variables
b1 <- ggplot(Baseball, aes(x = League86, y = Logsalary)) +
  geom_boxplot(aes(fill = League86)) +
  labs(title = "League (1986)") +
  theme_bw() +
  theme(legend.position = "none")

b2 <- ggplot(Baseball, aes(x = Div86, y = Logsalary)) +
  geom_boxplot(aes(fill = Div86)) +
  labs(title = "Division (1986)") +
  theme_bw() +
  theme(legend.position = "none")

b3 <- ggplot(Baseball, aes(x = League87, y = Logsalary)) +
  geom_boxplot(aes(fill = League87)) +
  labs(title = "League (1987)") +
  theme_bw() +
  theme(legend.position = "none")

b4 <- ggplot(Baseball, aes(x = factor(cut(Yrs, c(0, 3, 6, 10, 25))),
                           y = Logsalary)) +
  geom_boxplot(fill = "lightgreen") +
  labs(title = "Years in Major Leagues", x = "Yrs") +
  theme_bw()

windows()
grid.arrange(b1, b2, b3, b4, ncol = 2, nrow = 2)

## Scatter Plot Matrix (部分重要變數)
windows()
ggpairs(Baseball,
        columns = c("Hit86", "Rb86", "Yrs", "Hitcr", "Runcr", "Logsalary"),
        aes(alpha = 0.5)) +
  theme_bw()


########################################################################
## Analysis Data
########################################################################

## 移除不能當作解釋變數的欄位：
##   Id, Name   : 識別用
##   Salary     : Logsalary = log(Salary)，放進去等於直接洩漏答案
##   Team86, Team87, Pos86 : 類別太多 (20 幾隊 / 20 幾種守位)，
##                           切分後測試資料可能出現訓練資料沒有的類別，
##                           線性迴歸會無法預測，因此主要分析不納入
##                           (與 CART_Baseball.r 中 fit1 的變數相同)
ball <- Baseball[, c("Bat86", "Hit86", "Hr86", "Run86", "Rb86", "Wlk86",
                     "Yrs", "Batcr", "Hitcr", "Hrcr", "Runcr", "Rbcr",
                     "Wlkcr", "League86", "Div86", "Puto86", "Asst86",
                     "Err86", "League87", "Logsalary")]

dim(ball)
str(ball)


########################################################################
## Training Data and Testing Data
########################################################################

set.seed(123)

split <- sample.split(
  ball$Logsalary,      # 注意：要放反應變數，不是整個資料框
  SplitRatio = 0.8
)

train_data <- subset(ball, split == TRUE)
test_data <- subset(ball, split == FALSE)

dim(train_data)
dim(test_data)


########################################################################
## (1) Multiple Linear Regression
########################################################################

lm.fit <- glm(
  Logsalary ~ .,
  family = gaussian(link = "identity"),
  data = train_data
)

summary(lm.fit)

## Prediction
pred.lm.fit <- predict(
  lm.fit,
  newdata = test_data
)

## Mean Squared Error
mse.lm.fit <- mean(
  (pred.lm.fit - test_data$Logsalary)^2
)
mse.lm.fit

## Root Mean Squared Error
rmse.lm.fit <- sqrt(mse.lm.fit)
rmse.lm.fit

## Predicted vs. Observed Values
data.frame(
  pred.Logsalary = pred.lm.fit,
  true.Logsalary = test_data$Logsalary
)

windows()
plot(
  pred.lm.fit,
  test_data$Logsalary,
  xlim = c(4, 8),
  ylim = c(4, 8),
  xlab = "Predicted Values",
  ylab = "Observed Values",
  main = "Multiple Linear Regression",
  pch = 19
)

abline(a = 0, b = 1, col = "red", lty = 2)


########################################################################
## (2) Regression Tree (CART)
########################################################################

set.seed(123)

tree.fit <- rpart(
  Logsalary ~ .,
  method = "anova",
  data = train_data
)

windows()
rpart.plot(tree.fit, extra = 101,
           main = "Regression Tree: Baseball")

## Variable Importance
tree.fit$variable.importance

## Complexity Parameter Table (交叉驗證誤差)
printcp(tree.fit)

windows()
plotcp(tree.fit)

## Pruning：選 xerror 最小的 cp 來修剪
best.cp <- tree.fit$cptable[which.min(tree.fit$cptable[, "xerror"]), "CP"]
best.cp

prune.fit <- prune(tree.fit, cp = best.cp)

windows()
rpart.plot(prune.fit, extra = 101,
           main = "Pruned Regression Tree: Baseball")

## Prediction (使用修剪後的樹)
pred.tree.fit <- predict(
  prune.fit,
  newdata = test_data
)

## Mean Squared Error
mse.tree.fit <- mean(
  (pred.tree.fit - test_data$Logsalary)^2
)
mse.tree.fit

## Root Mean Squared Error
rmse.tree.fit <- sqrt(mse.tree.fit)
rmse.tree.fit

## Predicted vs. Observed Values
data.frame(
  pred.Logsalary = pred.tree.fit,
  true.Logsalary = test_data$Logsalary
)

windows()
plot(
  pred.tree.fit,
  test_data$Logsalary,
  xlim = c(4, 8),
  ylim = c(4, 8),
  xlab = "Predicted Values",
  ylab = "Observed Values",
  main = "Regression Tree",
  pch = 19
)

abline(a = 0, b = 1, col = "red", lty = 2)


########################################################################
## (3) Random Forest Regression
########################################################################

set.seed(123)

rf.fit <- randomForest(
  Logsalary ~ .,
  data = train_data,
  ntree = 500,
  mtry = 6,          # 迴歸預設為 floor(p/3) = floor(19/3) = 6
  nodesize = 5,
  importance = TRUE
)

## Model Summary
print(rf.fit)

## OOB Mean Squared Error
tail(rf.fit$mse, 1)

## OOB Error Plot
windows()
plot(rf.fit,
     main = "Random Forest Regression: OOB MSE")

## Variable Importance
importance(rf.fit)

windows()
varImpPlot(rf.fit,
           main = "Variable Importance: Baseball (Random Forest)")

## Prediction
pred.rf.fit <- predict(
  rf.fit,
  newdata = test_data
)

## Mean Squared Error
mse.rf.fit <- mean(
  (pred.rf.fit - test_data$Logsalary)^2
)
mse.rf.fit

## Root Mean Squared Error
rmse.rf.fit <- sqrt(mse.rf.fit)
rmse.rf.fit

## Predicted vs. Observed Values
data.frame(
  pred.Logsalary = pred.rf.fit,
  true.Logsalary = test_data$Logsalary
)

windows()
plot(
  pred.rf.fit,
  test_data$Logsalary,
  xlim = c(4, 8),
  ylim = c(4, 8),
  xlab = "Predicted Values",
  ylab = "Observed Values",
  main = "Random Forest Regression",
  pch = 19
)

abline(a = 0, b = 1, col = "red", lty = 2)

## Effect of mtry on Random Forest
mtry.values <- 1:19
oob.mse <- numeric(length(mtry.values))

for (i in seq_along(mtry.values)) {

  set.seed(123)

  fit <- randomForest(
    Logsalary ~ .,
    data = train_data,
    ntree = 500,
    mtry = mtry.values[i]
  )

  oob.mse[i] <- tail(fit$mse, 1)
}

windows()
plot(mtry.values, oob.mse,
     type = "b",
     pch = 19,
     xlab = "Number of Variables (mtry)",
     ylab = "OOB Mean Squared Error",
     main = "Effect of mtry on Random Forest")

mtry.values[which.min(oob.mse)]   # OOB MSE 最小的 mtry


########################################################################
## (4) XGBoost Regression
########################################################################

## Convert Data to Numeric Matrix
## model.matrix 會把 factor (League86, Div86, League87) 轉成 0/1 變數
x.train <- model.matrix(
  Logsalary ~ . - 1,
  data = train_data
)

x.test <- model.matrix(
  Logsalary ~ . - 1,
  data = test_data
)

## Create DMatrix
dtrain <- xgb.DMatrix(
  data = x.train,
  label = train_data$Logsalary
)

dtest <- xgb.DMatrix(
  data = x.test,
  label = test_data$Logsalary
)

## XGBoost Parameters
params <- list(
  objective = "reg:squarederror",
  eval_metric = "rmse",
  max_depth = 3,
  eta = 0.05,
  subsample = 0.8,
  colsample_bytree = 0.8,
  tree_method = "hist"
)

## XGBoost Regression Model
set.seed(123)

xgb.fit <- xgb.train(
  params = params,
  data = dtrain,
  nrounds = 300,
  verbose = 0
)

## Model Summary
print(xgb.fit)

## Variable Importance
importance.xgb <- xgb.importance(
  model = xgb.fit
)

print(importance.xgb)

windows()
xgb.plot.importance(
  importance_matrix = importance.xgb,
  main = "Variable Importance: Baseball (XGBoost)"
)

## Prediction
pred.xgb.fit <- predict(
  xgb.fit,
  newdata = dtest
)

## Mean Squared Error
mse.xgb.fit <- mean(
  (pred.xgb.fit - test_data$Logsalary)^2
)
mse.xgb.fit

## Root Mean Squared Error
rmse.xgb.fit <- sqrt(mse.xgb.fit)
rmse.xgb.fit

## Predicted vs. Observed Values
data.frame(
  pred.Logsalary = pred.xgb.fit,
  true.Logsalary = test_data$Logsalary
)

windows()
plot(
  pred.xgb.fit,
  test_data$Logsalary,
  xlim = c(4, 8),
  ylim = c(4, 8),
  xlab = "Predicted Values",
  ylab = "Observed Values",
  main = "XGBoost Regression",
  pch = 19
)

abline(a = 0, b = 1, col = "red", lty = 2)

## Effect of max_depth on XGBoost (用訓練資料切出驗證資料)
depth.values <- 1:8
validation.rmse <- numeric(length(depth.values))

set.seed(123)
val.index <- sample(
  seq_len(nrow(x.train)),
  size = floor(0.2 * nrow(x.train))
)

dtrain.inner <- xgb.DMatrix(
  data = x.train[-val.index, , drop = FALSE],
  label = train_data$Logsalary[-val.index]
)

dvalid <- xgb.DMatrix(
  data = x.train[val.index, , drop = FALSE],
  label = train_data$Logsalary[val.index]
)

for (i in seq_along(depth.values)) {

  params.temp <- params
  params.temp$max_depth <- depth.values[i]

  set.seed(123)

  fit.temp <- xgb.train(
    params = params.temp,
    data = dtrain.inner,
    nrounds = 300,
    verbose = 0
  )

  pred.temp <- predict(
    fit.temp,
    newdata = dvalid
  )

  validation.rmse[i] <- sqrt(mean(
    (pred.temp - train_data$Logsalary[val.index])^2
  ))
}

windows()
plot(
  depth.values,
  validation.rmse,
  type = "b",
  pch = 19,
  xlab = "Maximum Tree Depth",
  ylab = "Validation RMSE",
  main = "Effect of max_depth on XGBoost"
)


########################################################################
## Model Comparison
## Linear Regression vs. CART vs. Random Forest vs. XGBoost
########################################################################

## MSE and RMSE Comparison (Logsalary 尺度)
results <- data.frame(
  Model = c("Linear Regression", "Regression Tree",
            "Random Forest", "XGBoost"),
  MSE = c(mse.lm.fit, mse.tree.fit, mse.rf.fit, mse.xgb.fit),
  RMSE = c(rmse.lm.fit, rmse.tree.fit, rmse.rf.fit, rmse.xgb.fit)
)

## 換算回原始薪資尺度 (千美元) 的 RMSE，比較好解釋
true.salary <- exp(test_data$Logsalary)
results$RMSE.Salary <- c(
  sqrt(mean((exp(pred.lm.fit)   - true.salary)^2)),
  sqrt(mean((exp(pred.tree.fit) - true.salary)^2)),
  sqrt(mean((exp(pred.rf.fit)   - true.salary)^2)),
  sqrt(mean((exp(pred.xgb.fit)  - true.salary)^2))
)

results <- results[order(results$MSE), ]   # 依 MSE 由小到大排序
print(results, row.names = FALSE)

## Bar Chart of Test MSE
windows()
ggplot(results, aes(x = reorder(Model, MSE), y = MSE, fill = Model)) +
  geom_col() +
  geom_text(aes(label = round(MSE, 3)), vjust = -0.4) +
  labs(title = "Test MSE Comparison", x = "Model", y = "Test MSE") +
  theme_bw() +
  theme(legend.position = "none")

## Visualization of Prediction Results
make.plot <- function(pred, title, col) {
  ggplot(
    data.frame(
      Predicted = pred,
      Observed = test_data$Logsalary
    ),
    aes(x = Predicted, y = Observed)
  ) +
    geom_point(color = col) +
    geom_abline(
      intercept = 0,
      slope = 1,
      color = "red",
      linetype = 2
    ) +
    coord_fixed(xlim = c(4, 8), ylim = c(4, 8)) +
    labs(title = title) +
    theme_bw()
}

p1 <- make.plot(pred.lm.fit,   "Linear Regression", "blue")
p2 <- make.plot(pred.tree.fit, "Regression Tree",   "purple")
p3 <- make.plot(pred.rf.fit,   "Random Forest",     "darkgreen")
p4 <- make.plot(pred.xgb.fit,  "XGBoost",           "darkorange")

windows()
grid.arrange(p1, p2, p3, p4, ncol = 2, nrow = 2)


########################################################################
## 補充：加入 Team86、Pos86、Team87 等多類別變數的樹模型
## (樹模型可處理多類別變數；與 CART_Baseball.r 的 fit2 對應)
########################################################################

ball.full <- Baseball[, !(names(Baseball) %in% c("Id", "Name", "Salary"))]

train_full <- subset(ball.full, split == TRUE)
test_full <- subset(ball.full, split == FALSE)

set.seed(123)

tree.full <- rpart(
  Logsalary ~ .,
  method = "anova",
  data = train_full
)

windows()
rpart.plot(tree.full, extra = 101,
           main = "Regression Tree (With Team / Position)")

tree.full$variable.importance

pred.tree.full <- predict(tree.full, newdata = test_full)
mse.tree.full <- mean((pred.tree.full - test_full$Logsalary)^2)
mse.tree.full


########################################################################
## 重要參數說明
########################################################################
# Logsalary          反應變數 = log(Salary)，薪資右偏，取 log 後較對稱
# sample.split()     依反應變數切分 80% 訓練 / 20% 測試
#
# [CART]
# method = "anova"   迴歸樹
# printcp / plotcp   各 cp 值的交叉驗證誤差 (xerror)
# prune()            以 xerror 最小的 cp 修剪，避免過度配適
#
# [Random Forest]
# ntree = 500        建立 500 棵決策樹
# mtry = 6           每次節點分裂隨機選取 6 個候選變數 (迴歸預設 p/3)
# nodesize = 5       終端節點的最小樣本數
# importance = TRUE  計算變數重要性 (%IncMSE, IncNodePurity)
#
# [XGBoost]
# nrounds = 300      Boosting 次數
# max_depth = 3      每棵樹最大深度
# eta = 0.05         學習率
# subsample = 0.8    每次抽樣 80% 的樣本
# colsample_bytree = 0.8  每棵樹抽樣 80% 的變數
########################################################################
