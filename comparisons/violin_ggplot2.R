library(ggplot2); library(ggsci)
d <- read.csv("violin_data.csv"); d$group <- factor(d$group)
n <- table(d$group); lab <- paste0(names(n), "\n(n=", as.integer(n), ")")
set.seed(0)
p <- ggplot(d, aes(group, y, fill = group)) +
  geom_violin(trim = FALSE, alpha = 0.72, colour = "grey20", linewidth = 0.4) +
  geom_boxplot(width = 0.1, fill = "white", outlier.shape = NA, colour = "grey20", linewidth = 0.4) +
  geom_jitter(aes(colour = group), width = 0.08, shape = 21, fill = NA, size = 1.5, stroke = 0.5, alpha = 0.8) +
  scale_fill_npg() + scale_colour_npg() + scale_x_discrete(labels = lab) +
  labs(x = "group", y = "y", caption = sprintf("violin (trim=FALSE) + box: median/IQR; n=%d", nrow(d))) +
  theme_bw(base_size = 9) + theme(legend.position = "none", plot.caption = element_text(hjust = 0.5, size = 7))
ggsave("violin_ggplot2.png", p, width = 89, height = 80, units = "mm", dpi = 300)
