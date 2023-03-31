import operator
import zipfile


class UpdateableZipFile(zipfile.ZipFile):
    """A patched version of Python's zipfile that supports removing files in 'a' mode

    Python does not implement this although the issue has been
    known for over a decade and a fix exists:
    https://github.com/python/cpython/pull/19358

    This class ports the CPython fix forward.
    """

    def remove(self, member):
        """Remove a file from the archive. The archive must be open with mode 'a'"""

        if self.mode != "a":
            raise RuntimeError("remove() requires mode 'a'")
        if not self.fp:
            raise ValueError("Attempt to write to ZIP archive that was already closed")
        if self._writing:
            raise ValueError(
                "Can't write to ZIP archive while an open writing handle exists."
            )

        # Make sure we have an info object
        if isinstance(member, zipfile.ZipInfo):
            # 'member' is already an info object
            zinfo = member
        else:
            # get the info object
            zinfo = self.getinfo(member)

        return self._remove_member(zinfo)

    def _remove_member(self, member):
        # get a sorted filelist by header offset, in case the dir order
        # doesn't match the actual entry order
        fp = self.fp
        entry_offset = 0
        filelist = sorted(self.filelist, key=operator.attrgetter("header_offset"))
        for i in range(len(filelist)):
            info = filelist[i]
            # find the target member
            if info.header_offset < member.header_offset:
                continue

            # get the total size of the entry
            entry_size = None
            if i == len(filelist) - 1:
                entry_size = self.start_dir - info.header_offset
            else:
                entry_size = filelist[i + 1].header_offset - info.header_offset

            # found the member, set the entry offset
            if member == info:
                entry_offset = entry_size
                continue

            # Move entry
            # read the actual entry data
            fp.seek(info.header_offset)
            entry_data = fp.read(entry_size)

            # update the header
            info.header_offset -= entry_offset

            # write the entry to the new position
            fp.seek(info.header_offset)
            fp.write(entry_data)
            fp.flush()

        # update state
        self.start_dir -= entry_offset
        self.filelist.remove(member)
        del self.NameToInfo[member.filename]
        self._didModify = True

        # seek to the start of the central dir
        fp.seek(self.start_dir)
