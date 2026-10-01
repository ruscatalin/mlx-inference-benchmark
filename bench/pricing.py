"""Electricity price resolution, used at report time to compute €/Mtok."""


class PriceSource:
    def get(self, region):
        raise NotImplementedError


class StaticTable(PriceSource):
    def __init__(self, pricing_yaml):
        self.pricing_yaml = pricing_yaml

    def get(self, region):
        regions = self.pricing_yaml.get("regions", {})
        if region not in regions:
            raise ValueError(f"unknown region {region!r}; known: {sorted(regions)}")
        return regions[region]


class WebSource(PriceSource):
    def get(self, region):
        raise NotImplementedError


def resolve(args, pricing_yaml):
    if getattr(args, "price", None) is not None:
        currency = getattr(args, "currency", None) or pricing_yaml.get("currency", "EUR")
        return {"price_kwh": args.price, "currency": currency, "source": "user", "estimate": False}
    region = getattr(args, "region", None) or pricing_yaml.get("default_region")
    source_kind = getattr(args, "price_source", "static") or "static"
    source = StaticTable(pricing_yaml) if source_kind == "static" else WebSource()
    entry = source.get(region)
    return {"region": region, "price_kwh": entry["price_kwh"], "currency": entry.get("currency", "EUR"),
            "source": entry.get("source", "static"), "estimate": True}


CURRENCY_SYMBOL = {"EUR": "€", "USD": "$"}


def currency_symbol(code):
    return CURRENCY_SYMBOL.get(code, code)


def cost_per_mtok(price_kwh, tok_per_j):
    if not tok_per_j:
        return None
    return price_kwh / (tok_per_j * 3.6e6) * 1e6
