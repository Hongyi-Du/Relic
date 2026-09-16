"""Withheld contract: the silenced state of media elements is selectable.

Exercised only through ``soupsieve.select``, so any implementation that reports the state
correctly satisfies the contract regardless of how the state check is structured.
"""

from __future__ import annotations

from bs4 import BeautifulSoup

import soupsieve

MARKUP = """
<div>
  <video id="quiet_video" controls muted><source src="a.mp4" type="video/mp4"></video>
  <video id="loud_video" controls><source src="b.mp4" type="video/mp4"></video>
  <audio id="quiet_audio" controls muted><source src="c.ogg" type="audio/ogg"></audio>
  <audio id="loud_audio" controls><source src="d.ogg" type="audio/ogg"></audio>
  <div id="carrier" muted>not media</div>
</div>
"""


def selected_ids(selector: str, markup: str = MARKUP) -> list[str]:
    root = BeautifulSoup(markup, "html.parser").div
    assert root is not None
    return [element["id"] for element in soupsieve.select(selector, root)]


def test_silenced_state_matches_only_media_elements() -> None:
    assert selected_ids(":muted") == ["quiet_video", "quiet_audio"]


def test_silenced_state_narrows_to_the_requested_element_type() -> None:
    assert selected_ids("video:muted") == ["quiet_video"]
    assert selected_ids("audio:muted") == ["quiet_audio"]


def test_silenced_state_negates_like_any_other_state_check() -> None:
    assert selected_ids(":is(video, audio):not(:muted)") == ["loud_video", "loud_audio"]


def test_plain_attribute_presence_keeps_its_broader_meaning() -> None:
    assert selected_ids("[muted]") == ["quiet_video", "quiet_audio", "carrier"]
    assert selected_ids("div[muted]") == ["carrier"]
