# Image Processing GMM-algorithm-py
## Flow:
1 – chose two features from X digit data 
2 – chose 3 classes by random 
3 – get the data only two features and for three classes that you chose. 
4 – initialization (pi and mu and cov matrix)
5 – run GMM 100 iterations It’s more stable
### Step E:
#### Calculate  Pr(Xi|Zi=j)* Pr(Zi=j)  
#### result / result *sum(pi)

### Step M:
Updata pi and mu and cov 
6 – after we get pi mu and cov we can predict the Y:
7 – predicted we run on Wi,j and do max on every row and get the classes that give you the max value 
8 – calculate confusion matrix and Accuracy from Y to y predicted 
thon-code
