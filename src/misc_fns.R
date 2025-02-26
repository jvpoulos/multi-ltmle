######################
# Misc. functions    #
######################

# from weights package
dummify <- function(x, show.na=FALSE, keep.na=FALSE){
  if(!is.factor(x)){
    stop("variable needs to be a factor")
  }
  levels(x) <- c(levels(x), "NAFACTOR")
  x[is.na(x)] <- "NAFACTOR"
  levs <- levels(x)
  out <- model.matrix(~x-1)
  colnames(out) <- levs
  attributes(out)$assign <- NULL
  attributes(out)$contrasts <- NULL
  if(show.na==FALSE)
    out <- out[,(colnames(out)=="NAFACTOR")==FALSE]
  if(keep.na==TRUE)
    out[x=="NAFACTOR",] <- matrix(NA, sum(x=="NAFACTOR"), dim(out)[2])
  out
}

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

# Improved plotDAG function with LaTeX labels and proper spacing
plotDAG_improved <- function(DAG, tmax = NULL, xjitter = 0, yjitter = 0, 
                             node.action.color, vertex_attrs = list(), 
                             edge_attrs = list(), excludeattrs, customvlabs, 
                             verbose = getOption("simcausal.verbose"),
                             node_size = 20, label_size = 1.2, 
                             label_dist = 0, width = 800, height = 600,
                             use_latex = TRUE, arrow_size_mult = 1.5) {
  if (!requireNamespace("igraph", quietly = TRUE)) {
    stop("igraph package is required for this function to work. Please install igraph.",
         call. = FALSE)
  }
  
  # Check for latex2exp package if use_latex is TRUE
  if (use_latex && !requireNamespace("latex2exp", quietly = TRUE)) {
    warning("The latex2exp package is not available. Installing LaTeX formatting.")
    use_latex <- FALSE
  }
  
  set.seed(12345)
  DAG_orig <- DAG
  # Get a list of parent nodes
  par_nodes <- attr(DAG, "parents")
  latent.v <- attr(DAG, "latent.v")
  
  # ID attribute DAG nodes (those with missing order) and have them plotted separately
  ordervec <- sapply(DAG, '[[', "order")
  nullorder_idx <- which(sapply(ordervec, is.null))
  if (length(nullorder_idx) > 0) {
    DAG <- DAG[-nullorder_idx]
    class(DAG) <- "DAG"
    par_nodes_nullorder <- par_nodes[nullorder_idx]
    par_nodes <- par_nodes[-nullorder_idx]
  }
  
  # SUBSET A DAG by tmax
  if (!is.null(tmax)) {
    t_idx_all <- Nattr(DAG, "t") # a list of t values from current DAG (including NULLs)
    idx_tmax <- which(t_idx_all %in% tmax)
    if (length(idx_tmax) == 0) {
      warning("tmax argument could not be matched, using all available time points")
    } else {
      DAG <- DAG[1:max(idx_tmax)]
      par_nodes <- par_nodes[1:max(idx_tmax)]
      class(DAG) <- "DAG"
    }
  }
  
  # Remove nodes that are part of vector "excludeattrs":
  if (!missing(excludeattrs)) {
    excl.nodes <- names(par_nodes) %in% excludeattrs
    if (sum(excl.nodes) > 0) {
      par_nodes <- par_nodes[!excl.nodes]
      DAG <- DAG[!excl.nodes]
      class(DAG) <- "DAG"
    }
  }
  
  # Define x-axis coordinates - more space between nodes
  x_layout <- seq(0, length(par_nodes)*2.5, length.out = length(par_nodes))
  if (xjitter > 0) {
    x_layout <- x_layout + rnorm(n = length(x_layout), mean = 0, sd = xjitter)
  }
  
  # Plot when no t:
  if (is.null(DAG[[length(DAG)]]$t)) {
    arrow.width <- 0.8 * arrow_size_mult
    arrow.size <- 0.8 * arrow_size_mult
    vertsize <- node_size
    
    tmax <- 0
    num_bsl <- length(par_nodes)
    num_tv <- 0
    # More space vertically
    y_bsl <- rep(c(0,2,4), length.out = num_bsl)
    y_layout <- y_bsl
    
    # Plot when t nodes are present:
  } else {
    arrow.width <- 0.5 * arrow_size_mult
    arrow.size <- 0.5 * arrow_size_mult
    vertsize <- node_size
    y_vec <- NULL
    
    # get node t values in a DAG as a list (including NULLs)
    t_idx_all <- Nattr(DAG, "t") # a list of t values from current DAG (including NULLs)
    tunique <- unique(unlist(t_idx_all))
    t_idx_miss <- sapply(t_idx_all, is.null) # finding all nodes where t is undefined (null)
    t_idx_miss <- which(t_idx_miss %in% TRUE)
    
    # Improved y-coordinate spacing for time-varying nodes
    if (length(t_idx_miss) > 0) {
      y_vec <- c(y_vec, seq(0, length(t_idx_miss)*2.5, length.out = length(t_idx_miss)))
    }
    
    for (t in tunique) {
      idx_t <- which(t_idx_all %in% t)
      if (length(idx_t) > 0) {
        # More vertical space between time points
        y_vec <- c(y_vec, seq(0, length(idx_t)*2.5, length.out = length(idx_t)))
      }
    }
    y_layout <- y_vec
  }
  
  if (yjitter > 0) {
    y_layout <- y_layout + rnorm(n = length(y_layout), mean = 0, sd = yjitter)
  }
  layoutcustom_2 <- cbind(x_layout, y_layout)
  
  # create a layout for attribute nodes
  if (length(nullorder_idx) > 0) {
    gennames <- sapply(strsplit(names(par_nodes_nullorder), "_"), '[[', 1)
    excl <- rep(FALSE, length(par_nodes_nullorder))
    if (!missing(excludeattrs)) {
      excl <- gennames %in% excludeattrs
      if (sum(excl) > 0) {
        par_nodes_nullorder <- par_nodes_nullorder[!excl]
      }
    }
    x_layout_nullorder <- rep(-3, length(par_nodes_nullorder))  # Move further left
    maxy <- max(layoutcustom_2[,2])
    y_layout_nullorder <- seq(from=1, to=maxy, length.out=length(par_nodes_nullorder))
    layout_nullorder <- cbind(x_layout_nullorder, y_layout_nullorder)
    layoutcustom_2 <- rbind(layout_nullorder, layoutcustom_2)
    par_nodes <- c(par_nodes_nullorder, par_nodes)
  }
  
  # VERTEX AND EDGE plotting ATTRIBUTES with improved visibility
  attnames_ver <- names(vertex_attrs)
  attnames_edge <- names(edge_attrs)
  if (length(vertex_attrs) != 0 && (is.null(attnames_ver) || any(attnames_ver == ""))) {
    stop("please specify name for each attribute in vertex_attrs")
  }
  if (length(edge_attrs) != 0 && (is.null(attnames_edge) || any(attnames_edge == ""))) {
    stop("please specify name for each attribute in edge_attrs")
  }
  
  vertex_attrs_default <- list(
    color = NA, 
    label.color = "black", 
    shape = "circle", 
    size = vertsize, 
    label.cex = label_size, 
    label.dist = label_dist,
    frame.color = "lightgray",
    frame.width = 0.8
  )
  
  edge_attrs_default <- list(
    color = "gray40", 
    width = 0.7, 
    lty = 1, 
    arrow.width = arrow.width, 
    arrow.size = arrow.size
  )
  
  vertex_attrs <- append(vertex_attrs, vertex_attrs_default[!(names(vertex_attrs_default) %in% attnames_ver)])
  edge_attrs <- append(edge_attrs, edge_attrs_default[!(names(edge_attrs_default) %in% attnames_edge)])
  if (verbose) {
    message("using the following vertex attributes: "); message(vertex_attrs)
    message("using the following edge attributes: "); message(edge_attrs)
  }
  
  g <- igraph::graph.empty()
  vlabs <- names(par_nodes)
  g <- igraph::add.vertices(g, nv=length(vlabs), attr=vertex_attrs)
  igraph::V(g)$name <- vlabs
  
  for (i in c(1:length(names(par_nodes)))) {
    if (length(par_nodes[[i]]) > 0) {
      # check that parent nodes exist
      parents_i <- par_nodes[[i]]
      ind_parexist <- parents_i %in% (names(par_nodes))
      if (!all(ind_parexist)) {
        par_outvec <- parents_i[!ind_parexist]
        if (length(par_outvec) > 1) par_outvec <- paste0(par_outvec, collapse=",")
        if (verbose) {
          message("some of the extracted parent nodes weren't defined in the DAG and are being omitted: " %+% par_outvec)
        }
        parents_i <- parents_i[ind_parexist]
      }
      if (length(parents_i) > 0) {
        g <- igraph::add.edges(g, t(cbind(parents_i, names(par_nodes)[i])), attr=edge_attrs)
      }
    }
  }
  
  # mark intervention node labels with another color:
  if ("DAG.action" %in% class(DAG_orig)) {
    actnodenames <- attr(DAG_orig, "actnodes")
    if (missing(node.action.color)) node.action.color <- "red"
    igraph::V(g)[igraph::V(g)$name %in% actnodenames]$label.color <- node.action.color
  }
  
  # setting attributes for latent variables
  if (!is.null(latent.v)) {
    latent.idx <- which(names(par_nodes) %in% latent.v)
    
    # shift the y.grid for latent vars (U) by -0.25 of the minimum y value:
    min.y <- min(layoutcustom_2[, 2])
    max.y <- max(layoutcustom_2[, 2])
    layoutcustom_2[latent.idx, 2] <- min.y-0.25
    
    igraph::V(g)[igraph::V(g)$name %in% latent.v]$label.color <- "black"
    igraph::V(g)[igraph::V(g)$name %in% latent.v]$shape <- "circle"
    # igraph::E(g)[from(igraph::V(g)[igraph::V(g)$name %in% latent.v])]$lty <- 2 # dashed
    # to prevent a NOTE from using undeclared global "from" in igraph indexing
    Eg.sel <- eval(parse(text="igraph::E(g)[.from(igraph::V(g)[igraph::V(g)$name %in% latent.v])]$lty <- 5")) # long dash
  }
  
  # use user-supplied custom labels for DAG nodes:
  if (!missing(customvlabs)) {
    igraph::V(g)$name <- customvlabs
  }
  
  if (use_latex) {
    # Convert labels to LaTeX format using latex2exp
    latex_labels <- sapply(igraph::V(g)$name, function(lbl) {
      # Convert label to proper LaTeX format
      lbl_latex <- gsub("_([0-9]+)", "_{\\1}", lbl)  # Make subscripts
      lbl_latex <- gsub("\\^([0-9]+)", "^{\\1}", lbl_latex)  # Make superscripts
      # Return expression using latex2exp
      return(latex2exp::TeX(paste0("$", lbl_latex, "$")))
    })
    igraph::V(g)$label <- latex_labels
  }
  
  g <- igraph::set.graph.attribute(g, 'layout', layoutcustom_2)
  
  # Plot with appropriate margins
  par(mar=c(2, 2, 2, 2))  # Add some margin around the plot
  igraph::plot.igraph(g, asp=0.5)
  
  return(invisible(g))
}

## Other functions that may be needed for DAG plot (from simcausal)

N <- function(DAG) {
  if (!is.DAG(DAG) && !is.DAGnodelist(DAG)) {
    stop("Not a DAG object")
  }
  nodecount <- length(DAG)
  res <- seq_len(nodecount)
  class(res) <- "DAG.nodelist"
  ne <- new.env()
  assign("DAG", DAG, envir=ne)
  attr(res, "env") <- ne
  # the idea is to return the environment variable to avoid copying the DAG while subsetting
  # res
  # for now returning just the DAG itself
  attr(res, "env")$DAG
}
# select DAG nodes by t vector attribute
Ntvec <- function(DAG, tvec) {
  node_nms <- sapply(N(DAG), '[[', "name")
  # get actual t for each node and return only nodes that pass
  N_t_idx <- sapply(N(DAG), function(node) is.null(node[["t"]]) || (node[["t"]]%in%tvec))
  N_t <- N(DAG)[N_t_idx]
  class(N_t) <- "DAG.nodelist"
  N_t
}
# return a list of attribute values for a given attr name and list of nodes (DAG)
Nattr <- function(DAG, attr) {
  lapply(N(DAG), '[[', attr)
}

A <- function(DAG) {
  if (!is.DAG(DAG)) {
    stop("Not a DAG object")
  }
  res <- attributes(DAG)$actions
  if (is.null(res)) {
    NULL
  } else {
    # class(res) <- "DAG.action"
    res
  }
}

#' @import igraph

`%+%` <- function(a, b) paste0(a, b)	# custom cat function
`%/%` <- function(a, b) paste0(a, paste0(b, collapse=","))  # custom collapse with "," function
`.` <- function(...) return(unlist(list(...)))	# defining function . to avoid warnings from the R check

GetWarningsToSuppress <- function() {
  coercwarn <-  c("example warning 1.", "example warning 2.")
  return(coercwarn)
}

SuppressGivenWarnings <- function(expr, warningsToIgnore) {
  h <- function (w) {
    if (w$message %in% warningsToIgnore) invokeRestart( "muffleWarning" )
  }
  withCallingHandlers(expr, warning = h )
}

opts <- new.env(parent = emptyenv())
opts$vecfun <- NULL           # character vector of user-defined vectorized function names
opts$debug <- FALSE           # debug mode, when TRUE print all calls to dprint()

gvars <- new.env(parent = emptyenv())
gvars$misval <- NA_integer_ # the default missing value for observations
gvars$misXreplace <- 0L     # the default replacement value for misval that appear in the design matrix

vecfun.print <- function() {
  new <- opts$vecfun
  if (length(new)>1) new <- paste0(new, collapse=",")
  print("current list of user-defined vectorized functions: "%+%new)
  invisible(opts$vecfun)
}

vecfun.all.print <- function() {
  new <- opts$vecfun
  if (length(new)>1) new <- paste0(new, collapse=",")
  print("build-in vectorized functions:"); print(c(vector_ops_fcns, vector_math_fcns))
  print("user-defined vectorized functions: "%+%new)
  invisible(c(vector_ops_fcns,vector_math_fcns,new))
}

vecfun.add <- function(vecfun_names) { # Add vectorized function to global custom vectorized function list and return the old version
  old <- opts$vecfun
  opts$vecfun <- unique(c(opts$vecfun, vecfun_names))
  new <- opts$vecfun
  if (length(new)>1) new <- paste0(new, collapse=",")
  print("current list of user-defined vectorized functions: "%+%new)
  invisible(old)
}

vecfun.remove <- function(vecfun_names) { # Remove vectorized functions to global custom vectorized function list and return the old version
  old <- opts$vecfun
  idx_remove <- old%in%vecfun_names
  
  if (sum(idx_remove) < length(vecfun_names)) {
    fun_notfound <- vecfun_names[!(vecfun_names%in%old)]
    if (length(fun_notfound)>1) fun_notfound <- paste0(fun_notfound, collapse=",")
    warning("some of the function names in 'vecfun_names' were not found and cannot be removed: "%+%fun_notfound)
  }
  if (sum(idx_remove)>0) {
    opts$vecfun <- opts$vecfun[-(which(idx_remove))]
  }
  new <- opts$vecfun
  if (length(new)>1) new <- paste0(new, collapse=",")
  print("current list of user-defined vectorized functions: "%+%new)
  invisible(old)
}

vecfun.reset <- function() {
  old <- opts$vecfun
  opts$vecfun <- NULL
  invisible(old)
}
vecfun.get <- function() opts$vecfun
get_opts <- function() opts$debug # Return Current Debug Mode Setting
debug_set <- function() { # Set to Debug Mode
  mode <- TRUE
  old <- opts$debug
  opts$debug <- mode
  invisible(old)
}
debug_off <- function() { # Turn Off Debug Mode
  mode <- FALSE
  old <- opts$debug
  opts$debug <- mode
  invisible(old)
}
get_localvecfun <- function(DAG) attr(DAG, "vecfun")
dprint <- function(...) if (opts$debug) print(...) # debug-only version of print
is.DAG <- function(DAG) (("DAG" %in% class(DAG)) || ("DAG.action" %in% class(DAG))) # check its a DAG object
is.DAG.action <- function(DAG) ("DAG.action" %in% class(DAG)) # check its a DAG.action object
is.DAGnodelist <- function(DAG) "DAG.nodelist" %in% class(DAG) # check its a DAG object
is.node <- function(node) "DAG.node" %in% class(node) # check its a DAG.node object
is.Netnode <- function(node) "DAG.net" %in% class(node) # check its a DAG.net object (network node)

is.EFUP <- function(node) (!is.null(node$EFU)) # if the node is end follow-up when value = 1 (EFUP)

is.Bern <- function(node) {
  (node$distr%in%"Bern"||node$distr%in%"rbern") 		# if the node is Bernoulli random variable
}
is.attrnode <- function(node, DAG) {	# check if the node is an attribute (constant and cannot be intervened upon)
  attnames <- attr(DAG, "attnames")
  if (is.null(attnames)) {
    FALSE
  } else {
    gnodename <- as.character(unlist(strsplit(node$name, "_"))[1])
    gnodename%in%attr(DAG, "attnames")
  }
}
is.longfmt <- function(simdat) (attr(simdat, "dataform")%in%c("long")) # check data is in long format
is.LTCF <- function(simdat, outcome) {
  if (is.null(attr(simdat, "LTCF"))) {
    FALSE
  } else {
    attr(simdat, "LTCF")%in%outcome # check last time-point is carried forward (after failure)
  }
}
is.DAGlocked <- function(DAG) (!is.null(attr(DAG, "locked"))&&attr(DAG, "locked"))
get.actname <- function(DAG) attr(DAG, "actname")

print.DAG <- function(x, ...) {
  idx_nodes <- (!sapply(x, is.attrnode, x))
  x_notnull <- lapply(x, function(node) node[!sapply(node, is.null)])
  print(str(x_notnull[idx_nodes]))
  invisible(x)
}

print.DAG.action <- function(x, ...) {
  actnodes <- unique(attr(x, "actnodes"))
  lenact <- length(actnodes)
  if (lenact>1) {
    if (lenact>5) {
      actnodes <- actnodes[1]%+%", "%+%actnodes[2]%+%", ... , "%+%actnodes[lenact-1]%+%", "%+%actnodes[lenact]
    } else {
      actnodes <- paste0(actnodes, collapse=",")
    }
  }
  print("Action: "%+%attr(x, "actname"))
  print("ActionNodes: "%+%actnodes)
  print("ActionAttributes: "); print(attr(x, "attrs"))
  res <- list(
    Action=attr(x, "actname"),
    ActionNodes=attr(x, "actnodes"),
    ActionAttributes=attr(x, "attrs"))
  invisible(res)
}

print.DAG.node <- function(x, ...) str(x)

parents <- function(DAG, nodesChr) {
  parents_list <- attr(DAG, "parents")
  node_names <- names(parents_list)
  # print("node_names"); print(node_names)
  parents <- lapply(nodesChr, function(nodeChr) as.character(parents_list[[which(node_names%in%nodeChr)]]))
  names(parents) <- nodesChr
  parents
}

# Internal function that checks all DAG node objects have unique name attributes
check_namesunique <- function(inputDAG) {
  # node_names <- sapply(inputDAG, "[[", "name")
  node_names <- names(inputDAG)
  (length(unique(node_names))==length(inputDAG))
}

# Internal function that checks all DAG node objects are already in expanded format ((length(t)==1, length(order)==1))
check_expanded <- function(inputDAG) {
  for (node in inputDAG) {
    if (length(node$t)>1 | length(node$order)>1) {
      stop("node "%+% node$name %+% " is not expanded. Call constructor function node() first: node(name, t, distr, mean, probs, var, EFU, order)")
    }
  }
}