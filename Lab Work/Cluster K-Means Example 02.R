# Reading the data into R
cluster_ex02 = read.csv("Cluster Analysis Example 02.csv", header=TRUE)
fix(cluster_ex02)


# Standardize variables
mydata02 <- (cluster_ex02)[,-1]
mydata02 <- scale(mydata02) 
fix(mydata02)


# Identifying number of clusters
library(ggplot2)
library(factoextra)
library(cluster)

# Define a range of cluster numbers to evaluate
k_values <- 2:10

# Create a list to store silhouette widths for each k
sil_widths <- list()

# Loop over the range of cluster numbers
for (k in k_values) {
  # Perform K-means clustering
  km.res <- kmeans(mydata02, centers = k, nstart = 25)
  
  # Compute silhouette widths
  sil <- silhouette(km.res$cluster, dist(mydata02))
  
  # Store average silhouette width
  sil_widths[[as.character(k)]] <- mean(sil[, "sil_width"])
  
  # Optional: Visualize silhouette plot for current k
  # fviz_silhouette(sil) + ggtitle(paste("Silhouette Plot for k =", k))
}

# Convert list to a data frame for plotting
sil_widths_df <- data.frame(
  k = as.numeric(names(sil_widths)),
  avg_sil_width = unlist(sil_widths)
)

# Plot average silhouette width for each number of clusters
library(ggplot2)
ggplot(sil_widths_df, aes(x = k, y = avg_sil_width)) +
  geom_line() +
  geom_point() +
  ggtitle("Average Silhouette Width for Different Number of Clusters") +
  xlab("Number of Clusters") +
  ylab("Average Silhouette Width") +
  theme_minimal()


# Define a range of cluster numbers to evaluate
k_values <- 2:4

# Create a list to store silhouette plots
silhouette_plots <- list()

# Loop over the range of cluster numbers
for (k in k_values) {
  # Perform K-means clustering
  km.res <- kmeans(mydata02, centers = k, nstart = 25)
  
  # Compute silhouette widths
  sil <- silhouette(km.res$cluster, dist(mydata02))
  
  # Store silhouette plot
  silhouette_plots[[as.character(k)]] <- fviz_silhouette(sil) +
    ggtitle(paste("Silhouette Plot for k =", k))
}

# Display silhouette plots for different numbers of clusters
library(gridExtra)
do.call(grid.arrange, c(silhouette_plots, ncol = 2))


# K-Means Cluster Analysis
fit <- kmeans(mydata02, 2) # 2 cluster solution
names(fit)


# Cluster details
fit$cluster
fit$centers
fit$withinss
fit$tot.withinss

# Storing cluster membership

Cluster=fit$cluster
mydata02 <- data.frame(mydata02, Cluster) 
fix(mydata02)