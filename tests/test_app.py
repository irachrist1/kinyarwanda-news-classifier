from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_app_classifies_a_news_headline() -> None:
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py", default_timeout=15).run()
    assert not app.exception
    app.text_area[0].set_value("Abakinnyi b'umupira w'amaguru bitegura umukino w'igikombe cy'isi").run()
    app.button[0].click().run()
    assert not app.exception
    assert app.subheader[0].value
