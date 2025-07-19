# A bit overkill, I copy/pasted this from another project of mine



# TODOs:
#  - repetition by variable amounts
#  - load lazily from a text iterator
#  - Optional "max_length"s for Any, etc. to give some complexity guarantees

class Match:
    def __init__(self, start, stop, metadata={"extract": {}, "remove": {}}):
        self.start = start
        self.stop = stop
        self.metadata = {"extract": metadata.get("extract", {}).copy(), "remove": metadata.get("remove", {}).copy()}

    def __repr__(self):
        return f" [{self.start}, {self.stop}) "

def join_metadata(*matches):
    """
    Join metadata from multiple matches into a single match.
    """
    metadata = {"extract": {}, "remove": {}}
    for m in matches:
        for k, v in m.metadata["extract"].items():
            metadata["extract"][k] = v
        for k, v in m.metadata["remove"].items():
            metadata["remove"][k] = v
    return metadata

def join_metadata_dict(*matches):
    """
    Join metadata from multiple matches into a single match, returning a dictionary.
    """
    metadata = {"extract": {}, "remove": {}}
    for m in matches:
        for k, v in m["extract"].items():
            metadata["extract"][k] = v
        for k, v in m["remove"].items():
            metadata["remove"][k] = v
    return metadata

class Pattern:
    _parent = None
    _extract = False
    _extract_name = ""

    _remove = False

    def match(self, text):
        raise NotImplementedError("Subclasses must implement match method")

    def find(self, text, start=0, end=None, min_size=1):
        """Shortest matching interval at each valid starting location. Never yields empty intervals, so min_size is always at least 1."""
        # This is because any string has infinite non-overlapping empty intervals, which causes issues
        raise NotImplementedError("Subclasses must implement find method")

    def __add__(self, other):
        """Concatenate two patterns."""
        if isinstance(other, Pattern):
            return Concatenation(self, other)
        elif isinstance(other, str):
            return Concatenation(self, Literal(other))
        else:
            raise TypeError(f"Cannot concatenate {type(self)} with {type(other)}")

    def __radd__(self, other):
        """Concatenate two patterns. """
        if isinstance(other, Pattern):
            return Concatenation(other, self)
        elif isinstance(other, str):
            return Concatenation(Literal(other), self)
        else:
            raise TypeError(f"Cannot concatenate {type(other)} with {type(self)}")

    def __or__(self, other):
        """Disjunction of two patterns."""
        if isinstance(other, Pattern):
            return Disjunction(self, other)
        elif isinstance(other, str):
            return Disjunction(self, Literal(other))
        else:
            raise TypeError(f"Cannot disjoin {type(self)} with {type(other)}")

    def __ror__(self, other):
        """Disjunction of two patterns. """
        if isinstance(other, Pattern):
            return Disjunction(other, self)
        elif isinstance(other, str):
            return Disjunction(Literal(other), self)
        else:
            raise TypeError(f"Cannot disjoin {type(other)} with {type(self)}")

    def _get_toplevel(self):
        if self._parent is None:
            return self
        else:
            return self._parent._get_toplevel()

    def extract_preprocess(self, text):
        """Preprocess the text before extraction. Can be overridden by subclasses."""
        return text

    def extract(self, name="<anonymous>"):
        """
        Whenever the pattern matches, this part of the match will be extracted and returned in a tuple (or dictionary if as_dict=True) of strings.
        """
        if not self._extract and not self._remove:
            old_find = self.find
            self._extract_name = name
            def new_find(text, start=0, end=None, min_size=1):
                for m in old_find(text, start, end, min_size):
                    m.metadata["extract"][id(self)] = (name, self.extract_preprocess(text[m.start:m.stop]))
                    yield m
            self.find = new_find
            self._extract = True
        return self

    def get_extracted(self, text, start=0, end=None, min_size=1, as_dict=False):
        """
        Get the extracted text from the pattern.
        If the pattern is not an extractor, it will be converted to one.
        """
        for m in self.find(text, start, end, min_size):
            metadata = [x for x in m.metadata["extract"].values()]
            if not as_dict:
                yield tuple(x for _, x in metadata)
            else:
                ret = {}
                for k, v in metadata:
                    if k in ret:
                        if isinstance(ret[k], list):
                            ret[k].append(v)
                        else:
                            ret[k] = [ret[k], v]
                    else:
                        ret[k] = v
                yield ret

    def remove(self):
        """
        Whenever the pattern matches, this part of the match will be removed from the text.
        """
        if not self._extract and not self._remove:
            old_find = self.find
            def new_find(text, start=0, end=None, min_size=1):
                for m in old_find(text, start, end, min_size):
                    m.metadata["remove"][id(self)] = slice(m.start, m.stop)
                    yield m
            self.find = new_find
            self._remove = True
        return self

    def get_removed(self, text, start=0, end=None, min_size=1):
        """
        Get the removed text from the pattern.
        If the pattern is not a remover, it will be converted to one.
        """
        # TODO: make generator?
        chars = list(text)
        for m in self.find(text, start, end, min_size):
            for interval in m.metadata["remove"].values():
                start, stop = interval.start, interval.stop
                chars[start:stop] = [""] * (stop - start)  # Replace the removed text with empty strings
        return "".join(chars)


class Concatenation(Pattern):
    def __init__(self, p1, p2):
        self.p1 = p1
        self.p2 = p2
        p1._parent = self
        p2._parent = self
        super().__init__()

    def match(self, text):
        try:
            min_size = 1
            # Start with the smallest possible match for p1
            m1 = next(self.p1.find(text, 0, len(text), min_size))
            if m1.start != 0:
                return False
            while True:
                # Heuristic to deal with matching empty strings, since they're never yielded by "find"
                if m1.stop == len(text):
                    if self.p2.match(""):
                        return True
                    return False

                # Find the next match for p2 after m1
                m2 = next(self.p2.find(text, m1.stop, len(text)))
                min_size = m2.start
                # Then, find any matches that are long enough to fill the space between m1 and m2
                m1 = next(self.p1.find(text, 0, len(text), min_size)) # assert m1 gets bigger
                assert m1.stop - m1.start >= min_size, f"m1: {m1}, m2: {m2}, text: {text}"
                if m1.start != 0:
                    return False
                # If they are adjacent...
                if m1.stop == m2.start:
                    try:
                        # Then check that the second one can extend to the rest of the text...
                        m2_full = next(self.p2.find(text, m1.stop, len(text), len(text) - m2.start))
                        assert m2_full.stop == len(text) and m2_full.start == m2.start
                        # ...and if so, return True
                        return True
                    except StopIteration:
                        # Otherwise, we need to keep looking for a match; has to be strictly longer here so that we find a different m2 next time around
                        m1 = Match(m1.start, m1.stop + 1)
        except StopIteration:
            return False

    def find(self, text, start=0, end=None, min_size=1):
        if end is None:
            end = len(text)
        m1_starts = self.p1.find(text, start, end)
        p2_matches_empty = self.p2.match("")
        for m1 in m1_starts:
            if p2_matches_empty and m1.stop - m1.start >= min_size:
                # If p2 matches empty, we can yield the match immediately
                metadata = m1.metadata.copy()
                if self.p2._extract:
                    metadata["extract"][id(self.p2)] = (self.p2._extract_name, "")
                yield Match(m1.start, m1.stop, metadata=metadata)
                continue
            attempt_start = m1.start
            try:
                while True:
                    # Find the next match for p2 after m1
                    m2 = next(self.p2.find(text, m1.stop, end))
                    min_size_internal = m2.start - m1.start
                    # Then, find any matches that are long enough to fill the space between m1 and m2
                    m1 = next(self.p1.find(text, attempt_start, end, min_size_internal))
                    assert m1.stop - m1.start >= min_size_internal, f"m1: {m1} ({self.p1.__class__.__name__}), m2: {m2} ({self.p2.__class__.__name__}), min_size: {min_size_internal}"
                    if m1.start != attempt_start:
                        break
                    # If they are adjacent...
                    if m1.stop == m2.start:
                        try:
                            # Then check that the second one can extend to at least the minimum required size...
                            m2_full = next(self.p2.find(text, m1.stop, end, min_size - (m1.stop - m1.start)))
                            if m2_full.start == m1.stop:
                                # ...and if so, yield the match
                                yield Match(m1.start, m2_full.stop, metadata=join_metadata(m1, m2_full))
                                break
                            else:
                                m1 = Match(m1.start, m1.stop + 1)
                        except StopIteration:
                            # Otherwise, we need to keep looking for a match; has to be strictly longer here so that we find a different m2 next time around
                            m1 = Match(m1.start, m1.stop + 1)
            except StopIteration:
                continue

class Disjunction(Pattern):
    def __init__(self, p1, p2):
        self.p1 = p1
        self.p2 = p2
        p1._parent = self
        p2._parent = self
        super().__init__()

    def match(self, text):
        return self.p1.match(text) or self.p2.match(text)

    def find(self, text, start=0, end=None, min_size=1):
        i1 = self.p1.find(text, start, end, min_size)
        i2 = self.p2.find(text, start, end, min_size)

        m1 = next(i1, None)
        m2 = next(i2, None)
        while m1 is not None and m2 is not None:
            if m1.start < m2.start:
                yield m1
                m1 = next(i1, None)
            elif m2.start < m1.start:
                yield m2
                m2 = next(i2, None)
            else:
                # Yield only the shortest match from each starting point
                if m1.stop < m2.stop:
                    yield m1
                    m1 = next(i1, None)
                else:
                    yield m2
                    m2 = next(i2, None)
        while m1 is not None:
            yield m1
            m1 = next(i1, None)
        while m2 is not None:
            yield m2
            m2 = next(i2, None)

class Repetition(Pattern):
    def __init__(self, pattern, min_num=1, max_num=None):
        self.pattern = pattern
        self.min_num = min_num
        if max_num is None:
            self.max_num = float('inf')
        else:
            self.max_num = max_num

        pattern._parent = self
        super().__init__()

    def match(self, text):
        start = 0
        count = 0
        if text == "" and self.min_num == 0:
            return True
        while count < self.max_num:
            m = next(self.pattern.find(text, start), None)
            if m is None or m.start != start:
                return False
            if m.end == len(text) and count >= self.min_num:
                return True
            count += 1
            start = m.stop
        return False

    def find(self, text, start=0, end=None, min_size=1):
        if end is None:
            end = len(text)
        if min_size > end - start:
            return

        for m_start in self.pattern.find(text, start, end):
            count = 1
            start = m_start.stop
            metadata = m_start.metadata.copy()
            while count < self.max_num:
                m_next = next(self.pattern.find(text, start, end), None)
                if m_next is None or m_next.start != start:
                    break
                count += 1
                start = m_next.stop
                metadata = join_metadata_dict(metadata, m_next.metadata)
            if count >= self.min_num and start - m_start.start >= min_size:
                # print(text[m_start.start:start], m_start.start, start, metadata)
                yield Match(m_start.start, start, metadata=metadata)


class Literal(Pattern):
    def __init__(self, literal: str):
        self.literal = literal
        super().__init__()

    def match(self, text):
        return text == self.literal

    def find(self, text, start=0, end=None, min_size=1):
        if end is None:
            end = len(text)
        if min_size > len(self.literal):
            return
        start = text.find(self.literal, start, end)
        while start != -1:
            if end - start >= min_size:
                m = Match(start, start + len(self.literal))
                yield m
            start = text.find(self.literal, start + 1, end)

class Any(Pattern):
    def __init__(self):
        super().__init__()

    def match(self, text):
        return True

    def find(self, text, start=0, end=None, min_size=1):
        if end is None:
            end = len(text)
        if min_size > len(text):
            return
        for i in range(start, end - min_size + 1):
            yield Match(i, i + min_size)

class Chars(Pattern):
    def __init__(self, vocab=lambda x: True):
        self.vocab = vocab
        super().__init__()

    def match(self, text):
        return all(self.vocab(symbol) for symbol in text)

    def find(self, text, start=0, end=None, min_size=1):
        assert min_size >= 1, "Minimum size must be at least 1"
        if end is None:
            end = len(text)

        # print(text[start:end], start, end, min_size)

        if end - start < min_size:
            return
        current_start = None
        for idx in range(start, end):
            matches = self.vocab(text[idx])
            if current_start is None:
                if matches:
                    current_start = idx
            if current_start is not None:
                assert idx - current_start + 1 <= min_size, f"{idx}, {current_start}"
                if matches:
                    if idx - current_start + 1 == min_size:
                        yield Match(current_start, idx + 1)
                        current_start += 1
                else:
                    current_start = None


class Word(Pattern):
    def __init__(self, vocab=lambda x: x != " "):
        self.vocab = vocab
        assert not self.vocab(" ")
        super().__init__()

    def match(self, text):
        return len(text) >= 3 and text[0] == " " and text[-1] == " " and all(self.vocab(symbol) for symbol in text[1:-1])

    def find(self, text, start=0, end=None, min_size=1):
        if end is None:
            end = len(text)
        if end - start < min_size:
            return
        current_word_start = None
        for i in range(start, end):
            if text[i] == " ":
                if current_word_start is not None:
                    if i - current_word_start + 1 >= min_size and all([self.vocab(text[j]) for j in range(current_word_start + 1, i - 1)]):
                        yield Match(current_word_start, i+1)
                current_word_start = i

    def __add__(self, other):
        # make it so concatenating Word objects recognizes words with a single space between them, but still always a space on either side
        if isinstance(other, Word):
            return Words([self.vocab, other.vocab])
        elif isinstance(other, Words):
            return Words([self.vocab] + other.vocabs)
        else:
            return super().__add__(other)

    def extract_preprocess(self, text):
        """
        Preprocess the text before extraction.
        """
        return text.strip()

class Words(Pattern):
    def __init__(self, vocabs):
        raise NotImplementedError
        self.vocabs = vocabs
        super().__init__()

    def match(self, text):
        return len(text) >= 3 and text[0] == " " and text[-1] == " " and len(text.split(" ")) == len(self.vocabs) + 2 and all([all([self.vocabs[i](symbol) for symbol in section]) and section != "" for i, section in enumerate(text.split(" ")[1:-1])])

    def find(self, text, start=0, end=None, min_size=1):
        words = text.split(" ")
        if len(words) < 1 + len(self.vocabs):
            return
        for i, start_point in enumerate(words[1:len(words)-len(self.vocabs)+1]):
            if all([all([self.vocabs[i](symbol) for symbol in section]) and section != "" for i, section in enumerate(words[i:i+len(self.vocabs)])]):
                yield Match(start + sum(len(w) + 1 for w in words[:i]), start + sum(len(w) + 1 for w in words[:i + len(self.vocabs)])) #TODO: metadata extraction when fusing Word objects

    def __add__(self, other):
        if isinstance(other, Words):
            return Words(self.vocabs + other.vocabs)
        elif isinstance(other, Word):
            return Words(self.vocabs + [other.vocab])
        else:
            return super().__add__(other)

class Delimiters(Pattern):
    def __init__(self, open, inner_pat, close):
        self.open = open
        self.inner_pat = inner_pat
        self.close = close
        super().__init__()

    def match(self, text):
        return text.startswith(self.open) and text.endswith(self.close) and self.inner_pat.match(text[len(self.open):-len(self.close)]) and text.count(self.open) == text.count(self.close)

    def find(self, text, start=0, end=None, min_size=1):
        if end is None:
            end = len(text)

        open_pos_stack = []
        matches = []
        for i in range(start, end):
            if text.startswith(self.open, i):
                open_pos_stack.append(i)
            elif text.startswith(self.close, i) and open_pos_stack:
                start = open_pos_stack.pop()
                end = i + len(self.close)
                if end - start >= min_size:
                    inner_len = i - (start + len(self.open))
                    if inner_len == 0 and self.inner_pat.match(""):  # Handle empty inner pattern
                        metadata = {"extract": {}, "remove": {}}
                        if self.inner_pat._extract:
                            metadata["extract"][id(self.inner_pat)] = (self.inner_pat._extract_name, "")
                        matches.append(Match(start, end, metadata=metadata))
                    else:
                        inner_match = next(self.inner_pat.find(text, start + len(self.open), i, min_size=inner_len), None)
                        if inner_match is not None:
                            matches.append(Match(start, end, metadata=inner_match.metadata.copy()))
        matches.sort(key=lambda m: m.start)
        # TODO: make lazy
        for m in matches:
            yield m

# if __name__ == "__main__":
#     t = """theorem exists_lt_of_le_liminf [AddZeroClass α] [AddLeftStrictMono α] {x ε : α} {u : ℕ → α}
#         (hu_bdd : IsBoundedUnder GE.ge atTop u) (hu : x ≤ Filter.liminf u atTop) (hε : ε < 0) :
#     end ConditionallyCompleteLinearOrder
#
#         (hu : f.IsBoundedUnder (· ≤ ·) u := by isBoundedDefault) :
#         b ≤ limsup u f := by
#       revert hu_le
#       rw [← not_imp_not, not_frequently]
#
#
#
#     """
#     # TODO: no lines between theorems, "where" declarations, fails if last thing is a theorem
#     # pat = ((Literal("theorem") | "lemma" | "problem" | "def").extract("kind") + Word().extract("name") + Any() + ":=" + Any() + "\n" + Repetition(Literal("  ") + Any() + "\n", min_num=0) + "\n").extract("src")
#
#     pat = "\n" + Repetition(Literal("  ") + Any() + "\n", min_num=0).extract() + "\n"
#     for x in pat.get_extracted(t):
#         print(x)
#
#
#     block_comments = Delimiters("/-", Any(), "-/").remove()
#     inline_comments = (("--" + Any()).remove() + "\n")
#     t = block_comments.get_removed(t + "\n")
#     print(inline_comments.get_removed(t))

    # pat = Chars(lambda x: x != " ")
    # for x in pat.find("hello world", min_size=3):
    #     print(x)