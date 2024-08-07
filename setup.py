from setuptools import setup

setup(name='sequd',
      version='0.16',
      description='Hyperparameter Optimization based on Sequential Meta Machine Learning.',
      author='Zebin Yang and Aijun Zhang',
      author_email='yangzb2010@connect.hku.hk, ajzhang@umich.edu',
      license='BSD',
      packages=['sequd', 'sequd.pybayopt', 'sequd.pysequd', 'sequd.pybatdoe'], 
      install_requires=['joblib', 'numpy', 'pandas', 'scikit-learn', 'hyperopt', 
                        'pyunidoe @ git+https://github.com/kriap139/pyunidoe.git'], # 'smac==0.10.0', 'pyDOE', 'sobol_seq', 'tqdm', 'spearmint @ git+https://github.com/ZebinYang/spearmint-lite.git' 
      classifiers=['Development Status :: 4 - Beta',
                   'Environment :: Console',
                   'License :: OSI Approved :: BSD License',
                   'Operating System :: MacOS :: MacOS X',
                   'Operating System :: Microsoft :: Windows',
                   'Operating System :: POSIX',
                   'Programming Language :: Python'],
      zip_safe=False)
