FUNS <- new.env()
FUNS$SQR <- function(x) {x^2}
FUNS$CUBE <- function(x){x^3}
FUNS$SQRT <- function(x){ifelse(x<=0,.0001,sqrt(x))}#Set minimum value of .0001
FUNS$CURT <-function(x){x^.3333}
FUNS$LOG <- function(x){ifelse(x<=0,.0001,log(x))} #Set minimum value of .0001
FUNS$EXP <- function(x){ifelse(x<=0,.0001,exp(x))} #Set minimum value of .0001
FUNS$TAN <- function(x) {ifelse(x<=0,.0001,tan(x))} #Set minimum value of .0001
FUNS$SIN <- function(x) {ifelse(x<=0,.0001,sin(x))} #Set minimum value of .0001
FUNS$COS <- function(x) {ifelse(x<=0,.0001,cos(x))} #Set minimum value of .0001
FUNS$INV <- function(x) {ifelse(x<=0,.0001,1/x)} #Set minimum value of .0001
FUNS$SQI <- function(x) {ifelse(x<=0,.0001,1/(x^2))}#Set minimum value of .0001
FUNS$CUI <- function(x) {ifelse(x<=0,.0001,1/(x^3))}#Set minimum value of .0001
FUNS$SQRI <- function(x) {ifelse(x<=0,.0001,1/sqrt(x))}#Set minimum value of .0001
FUNS$CURI <- function(x) {ifelse(x<=0,.0001,1/(x^.3333))}#Set minimum value of .0001
FUNS$LOGI <- function(x) {ifelse(x<=0,.0001,1/log(x))}#Set minimum value of .0001
FUNS$EXPI <- function(x) {ifelse(x<=0,.0001,1/exp(x))}#Set minimum value of .0001
FUNS$TANI <- function(x) {ifelse(x<=0,.0001,1/tan(x))}#Set minimum value of .0001
FUNS$SINI <- function(x) {ifelse(x<=0,.0001,1/sin(x))}#Set minimum value of .0001
FUNS$COSI <- function(x) {ifelse(x<=0,.0001,1/cos(x))}#Set minimum value of .0001
