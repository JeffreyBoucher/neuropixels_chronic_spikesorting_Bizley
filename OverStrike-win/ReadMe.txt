
Purpose:
+ Overwrite noise with zeros in imec probe data.
Run messages are appended to OverStrike.log in the current working directory.

Usage:
>OverStrike -file=path/file.bin <-samps=a,b | -secs=a,b> [ options ] ### Did not understand option '-samps=255038,293334'.

Options:
-chans=0,3:5    ;only zero these channels (acquisition indices)
-linefill       ;rather than zeros, draw lines between segment endpoints

Notes:
- Use -file to set path and name of the .bin file.
- The .meta file must also exist in the same directory as the .bin.
- The named .bin is modified in place. The .meta is not changed.
- The .bin file length is not changed.
- The SY (sync) channel, if present, is never zeroed.
- The time span can be given in samples relative to file start.
- The time span can alternatively be given as seconds from file start.
- If -samps and -secs are both given, -secs is ignored.
- Ranges you enter are automatically bound to [0,file-length).
- Specify the whole file length as follows: -secs=0,1E9.
- OverStrike does one time span per call.
- Omit -chans=list to zero all channels (except SY).
- Use -chans=list to zero only selected channels.
- The channel indices must be original acquisition stream indices.
- AP indices are usually in range [0,383] and LF in range [384,767].
- Acquisition indices are displayed in the SpikeGLX FileViewer.
- Channel SY is not zeroed, even if in -chans list.


Change Log
----------
Version 1.1
- Support NP2020 quad-probes.
- Add -linefill option.

Version 1.0
- Initial release.


