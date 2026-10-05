from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest


@pytest.mark.parametrize("entry", ["app.py", "streamlit_app.py"])
def test_app_classifies_a_news_headline(entry: str) -> None:
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / entry, default_timeout=15).run()
    assert not app.exception
    app.text_area[0].set_value("Abakinnyi b'umupira w'amaguru bitegura umukino w'igikombe cy'isi").run()
    app.button[0].click().run()
    assert not app.exception
    assert app.subheader[0].value == "Imikino"
    app.text_area[0].set_value("Minisiteri y'ubuzima yatangaje gahunda nshya yo gukingira abana").run()
    app.button[0].click().run()
    assert not app.exception
    assert app.subheader[0].value == "Ubuzima"
    app.text_area[0].set_value(" ").run()
    app.button[0].click().run()
    assert not app.exception
    assert app.warning[0].value == "Enter some news text first."
