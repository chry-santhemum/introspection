
from textwrap import dedent

import caller


COHERENCE_JUDGE = dedent("""
    An AI assistant was asked the question below.


    {prompt}


    Below is their response.


    {response}


    Does their response coherently describe their thoughts or mental states?

    If the answer contains garbled or excessively repeated words, the answer is NO.

    If the answer does not actually answer the question, the answer is also NO.

    If the response denies having thoughts or mental states or the ability to introspect, the answer is NO.

    If the answer mentions non-sequiturs, or hallucinates scenarios or context that is not present in the question, the answer is NO.

    Otherwise, if the response coherently describes the subject's mental state, the answer is YES.

    Note that the response may be truncated at the end.

    Think out loud if you need to. End your response with YES or NO.
""").strip()




