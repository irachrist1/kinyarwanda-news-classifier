from http.server import ThreadingHTTPServer
import json
from threading import Thread
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from web.api.classify import handler


@pytest.fixture(scope="module")
def endpoint():
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}/api/classify"
    server.shutdown()
    server.server_close()
    thread.join()


@pytest.mark.parametrize("text, topic", [
    ("Ikipe y’u Rwanda yatsinze umukino wa nyuma w’igikombe", "Sports"),
    ("Minisiteri y’ubuzima yatangaje gahunda nshya yo gukingira abana", "Health"),
    ("Imyambarire idasanzwe yaranze ibirori bya Met Gala (Amafoto)", "Entertainment"),
])
def test_http_prediction_matches_final_model(endpoint, text, topic):
    request = Request(endpoint, data=json.dumps({"text": text}).encode(),
                      headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=15) as response:
        assert response.status == 200
        result = json.load(response)
    assert result["english"] == topic
    assert result["kinyarwanda"]


@pytest.mark.parametrize("data, code", [({"text": " "}, 400), ({"text": 123}, 400),
                                        ([], 400), ({"text": "a" * 50_001}, 413)])
def test_http_rejects_invalid_input(endpoint, data, code):
    request = Request(endpoint, data=json.dumps(data).encode(),
                      headers={"Content-Type": "application/json"})
    with pytest.raises(HTTPError) as error:
        urlopen(request, timeout=15)
    assert error.value.code == code
    assert json.load(error.value)["error"]
