import requests
from pyquery import PyQuery
import time
from tqdm import tqdm

def get_pyquery(url, call_count=0):
    html = requests.get(url, headers={'cookie': 'over18=1;'})
    if html.status_code == 200:
        dom = PyQuery(html.text, parser='html')
    else:    
        if call_count<5:
            time.sleep(3)
            dom = get_pyquery(url, call_count+1)
        else:
            raise Exception('failed at url:{}. error code:{}'.format(url, html.status_code))
    return dom

def get_home_index(board): #->int
    "get the latest page's index"
    url = 'https://www.ptt.cc/bbs/{board}/index.html'.format(board=board)
    dom = get_pyquery(url)    
    next_page_url = dom('.btn-group-paging a').eq(1).attr('href')
    index = re.findall('index[0-9]+', next_page_url)
    if not index:
        raise Exception('failed to find home index pattern: {}'.format(next_page_url))
    else:
        index = int(index[0][5:]) + 1
    return index

def get_post_from_index_page(url): #-> List[Tuple]
    'get all the post from a page'
    posts = []
    dom = get_pyquery(url)
    for dom_title in dom('.r-ent').items():
        hrefs = dom_title('a')
        if hrefs:
            text = hrefs.eq(0).text()
            href = hrefs.eq(0).attr('href')  
            author = dom_title('.meta .author').text()
            posts.append((text, href, author))
    return posts
    
def get_raw_pushes_list_from_post(post_url, ignore_error=True):
    try:
        dom = get_pyquery(post_url)    
    except:
        print('failed at {}'.format(post_url))
        return []
    pushes = []
    dom_pushes = dom('.push')
    for push in dom_pushes.items():
        zip_ = [i.text() for i in push('span').items()]
        if len(zip_) < 4:
            # not a normal push
            continue
        else:
            push_type, user, content, date = zip_
            content = content[2:]
            pushes.append((push_type, user, content, date))
    return pushes
        
def combine_user_push(raw_pushes): #-> List
    'combined pushes that been split due to length limit.'
    merged_pushes = []
    prev_user = None
    prev_push_type = None
    prev_content = ''
    prev_date = None
    for push_type, user, content, date in raw_pushes:            
        if user != prev_user:
            if prev_user is not None:
                # not first iter
                val = (prev_user, prev_push_type, prev_content, prev_date)
                merged_pushes.append(val)
            prev_user = user
            prev_push_type = push_type
            prev_content = content
            prev_date = date
            
        else:
            prev_content = prev_content + content
    return merged_pushes
        
def get_posts_from_board(board, start=None, end=None, n_from_latest=None):
    base_index_url = 'https://www.ptt.cc/bbs/{}/index{}.html'
    if n_from_latest:
        # set start end by latest index
        start = get_home_index(board)
        end = start - n_from_latest + 1
    if start < end:
        start, end = end, start
    all_posts = []
    for no in tqdm(range(start, end-1, -1)):
        url = base_index_url.format(board, no)
        cur_posts = get_post_from_index_page(url)
        if cur_posts:
            all_posts.extend(cur_posts)
    return all_posts
    