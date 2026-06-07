"""Monte Carlo race simulator — the core probability engine.

Consumes models from features/ratings/events/core/sequence/strategy and produces
a joint distribution over finishing positions. Every market probability is a
query against this distribution.
"""
