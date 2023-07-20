#	  MIT License
#    
#    Copyright (c) 2017 WindQAQ
#    
#    Permission is hereby granted, free of charge, to any person obtaining a copy
#    of this software and associated documentation files (the "Software"), to deal
#    in the Software without restriction, including without limitation the rights
#    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#    copies of the Software, and to permit persons to whom the Software is
#    furnished to do so, subject to the following conditions:
#    
#    The above copyright notice and this permission notice shall be included in all
#    copies or substantial portions of the Software.
#    
#    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#    SOFTWARE.
#    

"""Classes and operations related to processing data.
"""

import numpy as np

#--------------------------------------------------------------------------------------------------------------
def get_zip_file(url, filepath):
    """Gets zip file from url.

    Args:
        url: A string, the url of zip file.
        filepath: A string, the file path inside the zip file.

    Returns:
        A String, the content of wanted file.
    """

    from io import BytesIO
    from io import StringIO
    from zipfile import ZipFile
    import requests

    zipfile = ZipFile(BytesIO(requests.get(url).content))
    file = zipfile.open(filepath).read().decode('utf8')

    return StringIO(file)


class BatchGenerator(object):
    """Generator for data.
    """

    def __init__(self, x, y=None, batch_size=1024, shuffle=True):
        if y is not None and x.shape[0] != y.shape[0]:
            raise ValueError('The shape 0 of x should '
                             'be equal to that of y. ')

        self.x = x
        self.y = y
        self.length = x.shape[0]
        self.batch_size = batch_size
        self.shuffle = shuffle

    def next(self):
        start = end = 0
        length = self.length
        batch_size = self.batch_size

        if self.shuffle:
            permutation = np.random.permutation(length)
            self.x = self.x[permutation]
            self.y = self.y[permutation]

        flag = False
        while not flag:
            end += batch_size

            if end > length:
                end = length - 1
                flag = True

            yield self._get_batch(start, end)

            start = end

    def _get_batch(self, start, end):
        if self.y is not None:
            return self.x[start:end], self.y[start:end]
        else:
            return self.x[start:end]
