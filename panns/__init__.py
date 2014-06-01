#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2014 Liang Wang <liang.wang@cs.helsinki.fi>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html
#
# Liang Wang @ CS Dept, Helsinki Univ, Finland
# 2014.05.31
#


from .index import PannsIndex
from .utils import *

import logging
logger = logging.getLogger('panns')
logging.basicConfig(format='%(asctime)s : %(levelname)s : #%(process)d => %(message)s', level=logging.INFO)
