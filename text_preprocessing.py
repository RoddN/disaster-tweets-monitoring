from pyspark.sql import functions as F


def clean_text_expr(col_expr):
    c = F.lower(F.coalesce(col_expr, F.lit("")))
    c = F.regexp_replace(c, r"http\S+|https\S+|www\.\S+", " ")
    c = F.regexp_replace(c, r"@\w+", " ")
    c = F.regexp_replace(c, r"#(\w+)", "$1")
    c = F.regexp_replace(c, r"[^a-z0-9\s]", " ")
    c = F.regexp_replace(c, r"\s+", " ")
    return F.trim(c)
