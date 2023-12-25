library(tidyverse)
library(skimr)
library(DataExplorer)
library(caret)
library(pROC)
library(gbm)

# 加载数据
# 因变量分布
hist(database$t12,breaks = 50)

# 拆分数组，成为五组，做五折交叉验证
# 先分训练集测试集
trains<-createDataPartition(
  y=database$t12,
  p=0.8,
  list=F,
  times=1
)
data_train<-database[trains,]
data_test<-database[-trains,]

dvfunc<-dummyVars(~.,data=data_train[,4:124],fullRank = T)
data_trainx<-predict(dvfunc,newdata=data_train[,4:124])
data_trainy<-data_train$t12

data_testx<-predict(dvfunc,newdata=data_test[,4:124])
data_testy<-data_test$t12


#通过expand.grid函数设置模型的网格参数（如果要调参就设置调参的范围，不调就填一个值）
grid = expand.grid(
  n.trees = 4000,#最大迭代次数（最终模型中树的数量）
  shrinkage = 0.05, #学习率，默认为0.3
  interaction.depth = 6,#单个数的最大深度
  n.minobsinnode = 20 #开始分裂的节点中训练集样本的最小数目
)
grid

#使用caret包的trainControl函数定义5折交叉验证
library(caret)
cntrl = trainControl(
  method = "repeatedcv",
  number = 5,
  repeats = 1,
  verboseIter = TRUE,
  returnData = TRUE,
  returnResamp = "final",
  savePredictions = "final"
)

#使用caret包的train训练
train_gbm = train(
  x = data_trainx,
  y = data_trainy,
  trControl = cntrl,#设置交叉验证                                                                                                                                                         
  tuneGrid = grid,#网格参数
  method = "gbm"
)

result<-train_gbm$results #总的指标
foldresult<-train_gbm$resample #每一次每一折的指标
predrepeatedcv<-train_gbm$pred #这是所有次的交叉验证所有折的预测结果
fit_gbm_mod<-train_gbm$finalModel #这是得到的最终模型

# 需要整理一下预测的结果
predrepcv<-arrange(predrepeatedcv,rowIndex) #按rowIndex排序
predrepcv2<-predrepcv[,5:7] #取需要的列
predtrain<-aggregate(predrepcv2,by=list(rowIndex=predrepcv2$rowIndex),FUN=mean) #根据rowIndex分组求平均
predtrain<-data.frame(predtrain$pred) #求好的平均pred值，就是多次交叉验证得出的所有训练集的预测值
trainpred_data<-cbind(data_train,predtrain) #将训练集的预测值和原训练集数据组合起来

# 测试集预测结果
testpred<-predict(object = fit_gbm_mod,
                  newdata = data_testx,
                  n.trees = 4000,
                  type="link",
                  single.tree=FALSE)
# 测试集预测误差指标
testpredict<-defaultSummary(data.frame(obs=data_test$t12,
                                       pred=testpred))

testpred<-data.frame(testpred)
testpredict<-data.frame(testpredict)
testpred_data<-cbind(data_test,testpred) #将测试集的预测值和原测试集组合起来

# 给列改个名
colnames(testpred_data)[125]<-"pred"
colnames(trainpred_data)[125]<-"pred"
testpred_data$group<-"test"
trainpred_data$group<-"train"

# 组合起来就是总的数据集的预测值真实值对比
pred_data<-rbind(trainpred_data,testpred_data)

# 画图
predresult<-
  data.frame(obs=pred_data$t12,
             pred=pred_data$pred,
             group=pred_data$group
  )

ggplot(predresult,
       aes(x=obs,y=pred,color=group))+
  geom_point(shape=16,size=1.5,alpha=0.8)+
  geom_abline(intercept = 0,slope = 1,color="red",linetype=2)+
  geom_abline(intercept = 0.3,slope = 1,color="black",linetype=2)+
  geom_abline(intercept = -0.3,slope = 1,color="black",linetype=2)+
  scale_color_manual(values = c(train="blue",test="red"))+
  labs(fill=NULL,colour=NULL,
       x="Experiment",y="Predict",
       title = "GBM Experiment Compared with Predict")+
  theme_classic()+
  theme_bw()+
  theme(legend.position = "bottom",
        plot.title=element_text(hjust=0.5,size=16),
        axis.title=element_text(size=14))+
  coord_fixed(ratio=1)

# 保存数据
# 测试集预测误差指标
write.csv(testpredict,file = "testpredict.csv")
# 交叉验证误差指标
write.csv(result,file = "result.csv")
# 交叉验证各个折的误差指标
write.csv(foldresult,file = "foldresult.csv")
# 测试集预测结果
write.csv(testpred_data,file = "testpred_data.csv")
# 训练集预测结果
write.csv(trainpred_data,file = "trainpred_data.csv")
# 总的预测结果
write.csv(predrepeatedcv,file = "pred.csv")
# 没求平均前的交叉验证预测值
write.csv(predrepcv,file = "predrepcv.csv")
