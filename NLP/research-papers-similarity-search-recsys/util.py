import arxiv

# Construct the default API client.
client = arxiv.Client()

# Search for the 10 most recent articles matching the keyword "quantum."
search = arxiv.Search(
  query = "EMNLP",
  max_results = 10,
  sort_by = arxiv.SortCriterion.SubmittedDate
)

results = client.results(search)
for r in results:
  print(r)