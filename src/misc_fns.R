######################
# Misc. functions    #
######################

# function to bound probabilities to be used when making predictions
boundProbs <- function(x,bounds=c(0.025,1)){
  x[x>max(bounds)] <- max(bounds)
  x[x<min(bounds)] <- min(bounds)
  return(x)
}

# proper characters
proper <- function(s) sub("(.)", ("\\U\\1"), tolower(s), pe=TRUE)

# Summary figure for estimates
ForestPlot <- function(d, xlab, ylab){
  # Forest plot for summary figure
  p <- ggplot(d, aes(x=x, y=y, ymin=y.lo, ymax=y.hi,colour=forcats::fct_rev(Analysis))) + 
    geom_pointrange(size=1, position = position_dodge(width = -0.5)) + 
    coord_flip() +
    geom_hline(aes(yintercept=0), lty=2) +
    ylab(xlab) +
    xlab(ylab) #switch because of the coord_flip() above
  return(p)
}

gg_color_hue <- function(n) {
  hues = seq(15, 375, length = n + 1)
  hcl(h = hues, l = 65, c = 100)[1:n]
}

# survival plot (from simcausal)
plotSurvEst <- function (surv = list(), xindx = NULL, ylab = "", xlab = "t", 
          ylim = c(0, 1), legend.xyloc = "topright", ...){
  ptsize <- 1
  counter <- 0
  for (d.j in names(surv)) {
    counter <- counter + 1
    if (is.null(xindx)) 
      xindx <- seq(surv[[d.j]])
    plot(x = xindx, y = surv[[d.j]][xindx], col = counter, 
         type = "b", cex = ptsize, ylab = ylab, xlab = xlab, 
         ylim = ylim, ...)
    par(new = TRUE)
  }
  if(!is.null(legend.xyloc)){
    legend(legend.xyloc, legend = names(surv), col = c(1:length(names(surv))), 
           cex = ptsize, pch = 1)
  }
}

# function to calculate confidence intervals
CI <- function(est,infcurv,alpha=0.05){
  infcurv <- na.omit(infcurv)
  n <- length(infcurv)
  ciwidth <- qnorm(1-(alpha/2)) * sqrt(var(infcurv,na.rm = TRUE)/sqrt(n))
  CI <- c(est-ciwidth, est+ciwidth)
  return(CI)
}