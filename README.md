# Path-Planning-Algo
This project contains an implementation of the Laplace Planning algorithm and RRT algorithm

## Notion Link

https://achieved-scissor-d10.notion.site/1-11-Initial-Meeting-9f7a68183b324d34b60084f2e1bb4e17

## Colab

https://colab.research.google.com/drive/1K6e7tHyN4_U_oqhsQZbIC9RmDX5Kj38b?authuser=3

## Notes

For RRT, I've been thinking about modifying the probability of how random points are chosen. \
Runtime Comparison (excluding lines): RRT Connect < RRT Original < RRT Delete Branch

For Laplace, using OpenCV has significantly sped up the runtime, but it still won't work for world4.png.
