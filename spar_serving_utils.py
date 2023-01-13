"""
Several classes and methods to parse of the SPaR.txt output for further processing.
"""


class Indices(object):
    def __init__(self,
                 contiguous_start=-1,
                 contiguous_end=-1,
                 discontiguous_start=-1,
                 discontiguous_end=-1):
        self.ss = contiguous_start
        self.se = contiguous_end
        self.es = discontiguous_start
        self.ee = discontiguous_end

    def __str__(self):
        return str([self.ss, self.se, self.es, self.ee])

    def to_string(self):
        return str([self.ss, self.se, self.es, self.ee])

    def set_ss(self, idx):
        self.ss = idx

    def set_se(self, idx):
        self.se = idx

    def set_es(self, idx):
        self.es = idx

    def set_ee(self, idx):
        self.ee = idx

    def is_discontiguous(self):
        if self.se != self.ee and self.ee > 0:
            return True
        return False


class SingleSpan(object):
    """
    Single immutable span object
    """
    def __init__(self, span, span_type):
        # Text covered by span and type
        self.span = span
        self.span_type = span_type

    def __str__(self):
        if self:
            return self.span
        return ''

    def to_list(self):
        return [self.span, self.span_type]


class Sentence(object):
    """
    Single immutable sentence object
    """
    def __init__(self, sentence):
        self.sentence = sentence
        self.spans = {}

    def __str__(self):
        if self:
            return self.sentence
        return ''

    def to_dict(self):
        """ ToDO; might want to change how this object is formatted as a dict """
        return {self.sentence: self.spans}

    def add_span(self, span, indices):
        self.spans[span] = indices

    def get_indices(self, span):
        if span not in self.spans:
            raise ValueError("{} not in this sentence".format(span))
        return self.spans[span]


def mwe_list_to_string(mwe_list):
    """ Converts a list of tokens to a single word"""
    text = ''
    for w in mwe_list:
        if w.startswith('##'):
            text += w[2:]
        else:
            text += " " + w

    # ToDo ; decide if I would want to remove determiners here
    text = text[1:]
    # if text.startswith('the ') or text.startswith('The '):
    #     text = text[4:]
    # elif text.startswith('a ') or text.startswith('A '):
    #     text = text[2:]
    # elif text.startswith('an ') or text.startswith('An '):
    #     text = text[3:]
    return text


def update_indices(indices_obj:Indices):
    if indices_obj.se == -1:
        indices_obj.se = indices_obj.ss
    if indices_obj.es == -1:
        indices_obj.es = indices_obj.se
    if indices_obj.ee == -1:
        indices_obj.ee = indices_obj.es


def get_spans(word_list, tag_list, mwe_type):
    """
    Collects the MWE spans found by a trained tagger, handles discontiguous spans.
    """

    mwes = []
    indices = []
    types = []

    current_mwe = []
    current_indices = Indices()
    current_type = []

    previous_head = []
    previous_indices = Indices()
    just_removed = []

    for idx, t in enumerate(tag_list):
        # idx = t_id - 1
        t_head, current_type = t.split('-')

        if current_type != mwe_type:
            # We're collecting a specific type of MWE each time, e.g., objects / actions, so we can disentangle them
            if current_mwe:
                # store mwe and corresponding indices
                mwes.append(current_mwe)
                indices.append(current_indices)
                # In discontiguous cases, we want to store current_mwe into previous head
                if not just_removed:
                    # if we just encountered a BD-mwe_type, then previous_head should not change
                    previous_head = current_mwe
                    previous_indices = current_indices
            current_mwe = []
            current_indices = Indices()
            continue

        if t_head == 'BH':
            # We expect a completely new span of type `mwe_type'
            just_removed = []

            if not current_mwe:
                # Start collecting a new mwe
                current_mwe = [word_list[idx]]
                current_indices = Indices(idx)
            else:
                # Store old mwe and start collecting a new mwe
                mwes.append(current_mwe)
                indices.append(current_indices)
                previous_head = current_mwe
                previous_indices = current_indices
                current_mwe = [word_list[idx]]
                current_indices = Indices(idx)
        elif t_head == 'IH':
            # Continue collecting the same object, each time updating the head's end index
            # Due to tagging inaccuracy, this may continue from a BH or a BD
            current_mwe.append(word_list[idx])
            if current_indices.es == -1:
                current_indices.set_se(idx)
            else:
                current_indices.set_ee(idx)
        elif t_head == 'BD':
            # Case: BD-x with a preceding BH-x (note x is a given type)
            if just_removed != previous_head and previous_head in mwes:
                # Remove BH-based previous_head from mwes,
                mwes.reverse()        # ToDo; The last mwe may not be the mwe we want to remove ? otherwise simply pop()
                mwes.remove(previous_head)
                mwes.reverse()
                indices.reverse()
                indices.remove(previous_indices)
                indices.reverse()
                just_removed = previous_head
                #  and prepend it to the current_mwe, also setting the current indices to previous with new .es
                current_mwe = just_removed + [word_list[idx]]
                current_indices = previous_indices
                current_indices.set_es(idx)
                # reset previous head, considering that it has been used definitely now.
                previous_head = []

            else:   # Case: BD-x without a preceding BH-x (note x is a given type)
                # This CAN occur based on allowed transitions in the CRF, but shouldn't happen often
                if current_mwe:
                    # Store current mwe
                    mwes.append(current_mwe)
                    indices.append(current_indices)
                    previous_head = current_mwe
                    previous_indices = current_indices
                # start new mwe
                current_mwe = [word_list[idx]]
                current_indices = Indices(idx)
        elif t_head == 'ID':
            # Continue collecting discontiguous, each time updating the end's end index
            current_mwe.append(word_list[idx])
            current_indices.set_ee(idx)

    # if current_mwe:
    #     # store last mwe
    #     mwes.append(current_mwe)
    #     indices.append(current_indices)

    mwe_strings = [mwe_list_to_string(mwe) for mwe in mwes if mwe]
    [update_indices(i) for i in indices]
    return mwe_strings, indices


def parse_spar_output(prediction, span_types=['obj', 'act', 'func', 'dis']):
    """
    SPaR.txt outputs are formatted following the default AllenNLP json structure. This function grabs
    the spans from the output in text format.
    """
    # read predictions from file
    sentence = prediction["sentence"]
    sent_id = prediction["doc_id"]

    current_sent_obj = Sentence(sentence)

    mask = prediction['mask']
    tag_list = prediction['tags']
    token_list = prediction['words']

    predicted_tags = [t for m, t in zip(mask, tag_list) if m]
    for mwe_type in span_types:
        list_of_spans, list_of_indices_lists = get_spans(token_list, predicted_tags, mwe_type)
        for span, indices in zip(list_of_spans, list_of_indices_lists):
            current_span = SingleSpan(span, mwe_type)
            # store by indices, because spans may contain the same text
            current_sent_obj.spans[indices.to_string()] = current_span.to_list()

    # todo: clean these stitched together scripts at some point in my life :)
    sent_and_spans = current_sent_obj.to_dict()
    sentence = list(sent_and_spans.keys())[0]
    spans_dict = sent_and_spans[sentence]

    NER = {}
    for key in span_types:
        NER[key] = []

    for indices_str, span_tuple in spans_dict.items():
        # Indices, here, refers to the (possibly discontiguous) indices of the span in the sentence:
        # [start_start, start_end, end_start, end_end]
        ### todo; could use indices to count number of discontiguous spans
        # indices = [int(i) for i in indices_str[1:-1].split(',')]
        # assert len(indices) == 4
        span, span_type = span_tuple
        NER[span_type].append(span)

    return NER
