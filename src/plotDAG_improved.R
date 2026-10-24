# Plot of the simcausal DAG (Figure A1 of the paper): adapted from simcausal::plotDAG with LaTeX
# node labels (latex2exp) and a layout by time point.
# DAG: simcausal DAG; tmax: last time point shown; excludeattrs: nodes to omit; customvlabs: node
# labels; node_size, label_size, label_dist, arrow_size_mult, vertex_attrs, edge_attrs: igraph
# styling; mar: plot margins. Draws on the current device and returns the igraph object
# invisibly.
plotDAG_improved <- function(DAG, tmax = NULL, xjitter = 0, yjitter = 0,
                             node.action.color, vertex_attrs = list(),
                             edge_attrs = list(), excludeattrs, customvlabs,
                             verbose = getOption("simcausal.verbose"),
                             node_size = 20, label_size = 1.2,
                             label_dist = 0, width = 800, height = 600,
                             use_latex = TRUE, arrow_size_mult = 1.5, mar = c(2, 2, 2, 2)) {
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

  vertex_attrs <- append(vertex_attrs,
                         vertex_attrs_default[!(names(vertex_attrs_default) %in% attnames_ver)])
  edge_attrs <- append(edge_attrs,
                       edge_attrs_default[!(names(edge_attrs_default) %in% attnames_edge)])
  if (verbose) {
    message("using the following vertex attributes: "); message(vertex_attrs)
    message("using the following edge attributes: "); message(edge_attrs)
  }

  g <- igraph::make_empty_graph()
  vlabs <- names(par_nodes)
  g <- igraph::add_vertices(g, nv=length(vlabs), attr=vertex_attrs)
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
          message("some of the extracted parent nodes weren't defined in the DAG and are being ",
                  "omitted: ", par_outvec)
        }
        parents_i <- parents_i[ind_parexist]
      }
      if (length(parents_i) > 0) {
        g <- igraph::add_edges(g, t(cbind(parents_i, names(par_nodes)[i])), attr=edge_attrs)
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
    # long-dashed edges from latent variables
    Eg.sel <- eval(parse(text = paste0("igraph::E(g)[.from(igraph::V(g)[igraph::V(g)$name ",
                                       "%in% latent.v])]$lty <- 5")))
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

  g <- igraph::set_graph_attr(g, 'layout', layoutcustom_2)

  par(mar = mar)
  igraph::plot.igraph(g, asp=0.5)

  return(invisible(g))
}

## Helper functions from simcausal used by plotDAG_improved()

# list of the DAG's nodes
N <- function(DAG) {
  if (!is.DAG(DAG) && !is.DAGnodelist(DAG)) {
    stop("Not a DAG object")
  }
  nodecount <- length(DAG)
  res <- seq_len(nodecount)
  class(res) <- "DAG.nodelist"
  ne <- new.env()
  assign("DAG", DAG, envir = ne)
  attr(res, "env") <- ne
  attr(res, "env")$DAG
}

# attribute `attr` of every node of the DAG
Nattr <- function(DAG, attr) {
  lapply(N(DAG), "[[", attr)
}

# class checks for DAG objects
is.DAG <- function(DAG) (("DAG" %in% class(DAG)) || ("DAG.action" %in% class(DAG)))
is.DAGnodelist <- function(DAG) "DAG.nodelist" %in% class(DAG)
