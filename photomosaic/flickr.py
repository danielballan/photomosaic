import os
import re
import urllib
import requests
import itertools
from tqdm import tqdm
from .photomosaic import options


PUBLIC_URL = "https://www.flickr.com/photos/"
API_URL = 'https://api.flickr.com/services/rest/'
PATH = "http://farm{farm}.staticflickr.com/{server}/"
NAME = "{id}_{secret}_b.jpg"


def _flickr_request(**kwargs):
    params = dict(api_key=options['flickr_api_key'],
                  format='json',
                  nojsoncallback=1,
                  **kwargs)
    response = requests.get(API_URL, params=params)
    return response.json()


def from_search(text, dest, cutoff=None):
    """
    Download photos matching a search query.

    Parameters
    ----------
    text : string
        Search query
    dest : string
        Output directory
    cutoff : integer or None, optional
        Max number of images to download. By default, None; all matches
        up to Flickr's max (4000) will be downloaded.
    """
    os.makedirs(dest, exist_ok=True)
    total = itertools.count(0)
    for page in itertools.count(1):
        response = _flickr_request(
                method='flickr.photos.search',
                text=text,
                content_type=1,  # photos only
                page=page
        )
        if response.get('stat') != 'ok':
            # If we fail requesting page 1, that's an error. If we fail
            # requesting page > 1, we're just out of photos.
            if page == 1:
                raise RuntimeError("response: {}".format(response))
            break
        photos = response['photos']['photo']
        for photo in tqdm(photos, desc='downloading page {}'.format(page)):
            if (cutoff is not None) and (next(total) > cutoff):
                return
            url = (PATH + NAME).format(**photo)
            filename = (NAME).format(**photo)
            filepath = os.path.join(dest, filename)
            urllib.request.urlretrieve(url, filepath)


def _get_photoset(photoset_id, nsid, dest):
    os.makedirs(dest, exist_ok=True)
    for page in itertools.count(1):
        response = _flickr_request(
                method='flickr.photosets.getPhotos',
                photoset_id=photoset_id,
                nsid=nsid,
                content_type=1,  # photos only
                page=page
        )
        if response.get('stat') != 'ok':
            # If we fail requesting page 1, that's an error. If we fail
            # requesting page > 1, we're just out of photos.
            if page == 1:
                raise RuntimeError("response: {}".format(response))
            break
        photos = response['photoset']['photo']
        for photo in tqdm(photos, desc='downloading page {}'.format(page)):
            url = (PATH + NAME).format(**photo)
            filename = (NAME).format(**photo)
            filepath = os.path.join(dest, filename)
            urllib.request.urlretrieve(url, filepath)


def from_url(url, dest):
    """
    Download an album ("photoset") from its url.

    Parameters
    ----------
    url : string
        e.g., https://www.flickr.com/phtoos/<username>/sets/<photoset_id>
    dest : string
        Output directory
    """
    m = re.match(PUBLIC_URL + "(.*)/sets/([0-9]+)", url)
    if m is None:
        raise ValueError("""Expected URL like:
https://www.flickr.com/photos/<username>/sets/<photoset_id>""")
    username, photoset_id = m.groups()
    response = _flickr_request(method="flickr.urls.lookupUser",
                               url=PUBLIC_URL + username)
    nsid = response['user']['username']['_content']
    return _get_photoset(photoset_id, nsid, dest)
