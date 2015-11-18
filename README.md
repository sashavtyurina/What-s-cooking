# What-s-cooking
STAT841 course project

Typical ingredients.
1. We build a background model of all the ingredients for all the recipes. (Corpus Q)
2. We also build a model for a chosen cuisine (italian, greek. ...) (Corpus P)
3. Using pointwise KLD (https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence) we identify the ingredients
that are typical for the chosen cuisine (italian, corpus P), but are not very frequent in the overall corpus ( Q ). 
