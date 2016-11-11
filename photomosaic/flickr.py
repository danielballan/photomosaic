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


def from_search(text, dest, cutoff=None, license=None):
    """
    Download photos matching a search query and the specified license(s).

    Parameters
    ----------
    text : string
        Search query
    dest : string
        Output directory
    cutoff : integer or None, optional
        Max number of images to download. By default, None; all matches
        up to Flickr's max (4000) will be downloaded.
    license : list or None
        List of license codes documented by Flickr at
        https://www.flickr.com/services/api/flickr.photos.licenses.getInfo.html
        If None, photomosaic defaults to ``[1, 2, 4, 5, 7, 8]``. See link for
        details.
    """
    if license is None:
        license = [1, 2, 4, 5, 7, 8]
    os.makedirs(dest, exist_ok=True)
    total = itertools.count(0)
    for page in itertools.count(1):
        response = _flickr_request(
                method='flickr.photos.search',
                license=','.join(map(str, license)),
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

    The is no programmatic license-checking here; that is up to the user.

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
