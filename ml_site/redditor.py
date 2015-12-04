import praw
import pdb
import time
import numpy as np

def redditor():

    r = praw.Reddit('PRAW related-question monitor by u/_Daimon_ v 1.0.'
                    'Url: https://praw.readthedocs.org/en/latest/'
                    'pages/writing_a_bot.html')
    prawWords = ['pentageon', 'oil']

    r.login('spy_bot007','motion')
    already_done = []
    comment_list = []
    subreddit = r.get_subreddit('worldnews')
    for submission in subreddit.get_hot(limit=1):
        op_text = submission.selftext.lower()
        has_praw = any(string in op_text for string in prawWords)
        flat_comments = praw.helpers.flatten_tree(submission.comments)
        already_done = set()
        for comment in flat_comments:
            if comment.id not in already_done:
                if hasattr(comment, 'body'):
                    comment_list.append(comment.body)
                    already_done.add(comment.id)



        # Test if it contains a PRAW-related question
        if submission.id not in already_done and has_praw:
            msg = '[PRAW related thread](%s)' % submission.short_link
            r.send_message('_motioncuty_', 'PRAW Thread', msg)
            already_done.append(submission.id)
            print msg

    print comment_list
    return np.array(comment_list)
