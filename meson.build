project(
  'pinball',
  'c', 'cpp', 'fortran',
  version: '0.1.0',
  meson_version: '>=1.1.0',
  default_options: [
    'buildtype=release',
  ],
)

py = import('python').find_installation(pure: false)

subdir('pinball')
subdir('fortran')
subdir('csrc')
