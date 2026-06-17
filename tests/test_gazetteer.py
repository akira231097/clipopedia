from clipopedia.retrieval.gazetteer import Gazetteer

GAZ = Gazetteer(
    guests=["Dr. Lena Ortiz", "Sam Whitfield", "Priya Raman"],
    hosts=["Mara Quinn", "Devin Cole"],
    shows=["The Long Game", "Builders & Backers"],
)


def test_resolves_dropped_honorific_and_case():
    assert GAZ.resolve_guests(["lena ortiz"]) == ["Dr. Lena Ortiz"]


def test_resolves_show_with_article_noise():
    assert GAZ.resolve_show("the long game podcast") == "The Long Game"


def test_unknown_entity_is_dropped():
    assert GAZ.resolve_guests(["Someone Entirely Unknown"]) == []


def test_dedupes_resolved_names():
    assert GAZ.resolve_guests(["lena ortiz", "Dr. Lena Ortiz"]) == ["Dr. Lena Ortiz"]


def test_host_resolution():
    assert GAZ.resolve_hosts(["mara quin"]) == ["Mara Quinn"]
