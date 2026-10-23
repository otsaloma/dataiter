# -*- coding: utf-8 -*-

# Copyright (c) 2025 Osmo Salomaa
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import pytest
import re

from dataiter import regex
from dataiter import Vector

class TestRegex:

    def test_contains(self):
        pattern = r"\(\d{4}\)"
        string = Vector(["Toy Story (1995)", "No year", ""])
        result = regex.contains(pattern, string)
        assert result.is_boolean()
        assert result.tolist() == [True, False, False]
        assert regex.contains(r"^$", string).tolist() == [False, False, False]
        assert regex.contains(r"story", string, flags=re.IGNORECASE).tolist() == [True, False, False]
        assert regex.contains(pattern, string[0]) is True
        assert string.re.contains(pattern).tolist() == [True, False, False]

    def test_extract(self):
        pattern = r"\((\d{4})\)|(none)"
        string = Vector(["Toy Story (1995)", "none", "Unknown", ""])
        assert regex.extract(pattern, string).tolist() == ["(1995)", "none", None, None]
        assert regex.extract(pattern, string, group=1).tolist() == ["1995", None, None, None]
        assert regex.extract(r"\((?P<year>\d{4})\)", string, group="year")[0] == "1995"
        assert regex.extract(pattern, string[0], group=1) == "1995"

    def test_findall(self):
        pattern = r"[a-z]"
        string = Vector(["asdf", "1234", ""])
        result = regex.findall(pattern, string)
        expected = [["a", "s", "d", "f"], [], None]
        assert result.tolist() == expected
        assert regex.findall(pattern, string[0]) == result[0]

    def test_fullmatch(self):
        pattern = r"[a-z]+"
        string = Vector(["asdf", "1234", ""])
        result = regex.fullmatch(pattern, string)
        assert isinstance(result[0], re.Match)
        assert result[1] is None
        assert result[2] is None
        match = regex.fullmatch(pattern, string[0])
        assert isinstance(match, re.Match)

    def test_match(self):
        pattern = r"[a-z]"
        string = Vector(["asdf", "1234", ""])
        result = regex.match(pattern, string)
        assert isinstance(result[0], re.Match)
        assert result[1] is None
        assert result[2] is None
        match = regex.match(pattern, string[0])
        assert isinstance(match, re.Match)

    def test_search(self):
        pattern = r"[a-z]"
        string = Vector(["asdf", "1234", ""])
        result = regex.search(pattern, string)
        assert isinstance(result[0], re.Match)
        assert result[1] is None
        assert result[2] is None
        match = regex.search(pattern, string[0])
        assert isinstance(match, re.Match)

    def test_split(self):
        pattern = r" +"
        string = Vector(["one two three", "four", ""])
        result = regex.split(pattern, string)
        expected = [["one", "two", "three"], ["four"], None]
        assert result.tolist() == expected
        assert regex.split(pattern, string[0]) == result[0]

    def test_sub(self):
        pattern = r"$"
        repl = "!"
        string = Vector(["great", "fantastic", ""])
        result = regex.sub(pattern, repl, string)
        expected = ["great!", "fantastic!", None]
        assert result.tolist() == expected
        assert regex.sub(pattern, repl, string[0]) == result[0]

    def test_sub_type_error(self):
        with pytest.raises(TypeError, match="Expected string, got int64"):
            regex.sub(r"$", "!", Vector([1, 2]))

    def test_subn(self):
        pattern = r"$"
        repl = "!"
        string = Vector(["great", "fantastic", ""])
        result = regex.subn(pattern, repl, string)
        expected = [("great!", 1), ("fantastic!", 1), None]
        assert result.tolist() == expected
        assert regex.subn(pattern, repl, string[0]) == result[0]
