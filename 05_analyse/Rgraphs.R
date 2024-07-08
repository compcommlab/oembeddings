# Load necessary libraries
library(ggplot2)
library(gridExtra)

# Read in the dataset
data <- read.csv("/Users/janabernhard/Documents/Projekte/2023_Embedding/analysis/validation_3_results_all_grouped.csv")  # Replace "your_dataset.csv" with the actual file path/name

# Define colors for different min counts
mincount_colors <- c("5" = "#0063A6", "10" = "#A71C49", "50" = "#F6A800", "100" = "#94C154")

# Function to calculate transparency based on window size
get_transparency <- function(window_size) {
  transparency_levels <- c(0.25, 0.6, 0.75, 0.9)
  return(transparency_levels[match(window_size, c(5, 6, 12, 24))])
}

# Function to create plot for each performance variable
create_plot <- function(data, column_name) {
  plot_title <- switch(
    column_name,
    "bestmatch" = "Intrinsic Task Semantic: Best Match",
    "opposite" = "Intrinsic Task Semantic: Opposite",
    "wordintrusion" = "Intrinsic Task Semantic: Word Intrusion",
    "mostsimilar" = "Intrinsic Task Syntactic: Most Similar",
    "ffp" = "Extrinsic Task: Author Classification",
    "topics" = "Extrinsic Task: Topic Classification",
    "sentiment" = "Extrinsic Task: Sentiment Classification"
  )
  
  y_limits <- switch(
    column_name,
    "bestmatch" = c(0.4, 0.7),
    "opposite" = c(0, 0.2),
    "wordintrusion" = c(0.8, 1),
    "mostsimilar" = c(0.5, 0.7),
    "ffp" = c(0.4, 0.6),
    "topics" = c(0.4, 0.6),
    "sentiment" = c(0.4, 0.6)
  )
  
  plot_cased <- ggplot(data[data$group_number %in% 2:17, ], 
                       aes(x = factor(group_number), y = !!sym(column_name), fill = factor(mincount), alpha = as.numeric(windows))) +
    geom_boxplot() +
    scale_fill_manual(values = mincount_colors, name = "Min Count") +
    scale_alpha_continuous(range = c(0.25, 0.9), guide = "none") +
    theme_minimal() +
    labs(x = "Group Number", y = "Performance") +
    coord_cartesian(ylim = y_limits) +  # Set y-axis limits
    ggtitle(paste("Cased Models")) +  # Add subtitle for cased models
    theme(legend.position = "bottom",
          axis.text.x = element_text(angle = 45, hjust = 1),
          panel.grid.major.x = element_blank(),
          panel.grid.minor.x = element_blank(),
          plot.title = element_text(hjust = 0.5))  # Center plot title
  
  plot_lowercased <- ggplot(data[data$group_number %in% 18:33, ], 
                            aes(x = factor(group_number), y = !!sym(column_name), fill = factor(mincount), alpha = as.numeric(windows))) +
    geom_boxplot() +
    scale_fill_manual(values = mincount_colors, name = "Min Count") +
    scale_alpha_continuous(range = c(0.25, 0.9), guide = "none") +
    theme_minimal() +
    labs(x = "Group Number", y = "Performance") +
    coord_cartesian(ylim = y_limits) +  # Set y-axis limits
    ggtitle(paste("Lowercased Models")) +  # Add subtitle for lowercased models
    theme(legend.position = "bottom",
          axis.text.x = element_text(angle = 45, hjust = 1),
          panel.grid.major.x = element_blank(),
          panel.grid.minor.x = element_blank(),
          plot.title = element_text(hjust = 0.5))  # Center plot title
  
  return(list(plot_cased, plot_lowercased))
}

# Loop through each performance variable, create plot, and save as SVG/PDF
performance_variables <- c("bestmatch", "opposite", "wordintrusion", "mostsimilar", "ffp", "topics", "sentiment")
for (variable in performance_variables) {
  plots <- create_plot(data, variable)
  filename <- paste0("/Users/janabernhard/Documents/Projekte/2023_Embedding/analysis/R_graphs/plot_", variable, "_pergroup")
  ggsave(paste0(filename, ".svg"), arrangeGrob(plots[[1]], plots[[2]], nrow = 1), device = "svg", width = 210, height = 148, units = "mm")
  ggsave(paste0(filename, ".pdf"), arrangeGrob(plots[[1]], plots[[2]], nrow = 1), device = "pdf", width = 210, height = 148, units = "mm")
}