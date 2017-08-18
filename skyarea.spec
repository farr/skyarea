Summary: Compute credible regions on the sky from RA-DEC MCMC samples
Name: skyarea
Version: 0.3.2
Release: 1%{?dist}
Source: https://github.com/farr/skyarea/archive/v%{version}/%{name}-%{version}.tar.gz
License: MIT
Group: Development/Libraries
BuildArch: noarch
Vendor: Will M. Farr <will.farr@ligo.org>
Packager: Leo Singer <leo.singer@ligo.org>
Requires: numpy scipy python-six healpy glue lalinference-python >= 1.9.4
Url: http://farr.github.io/skyarea/
BuildRequires: python-setuptools

%description
Computing credible areas and p-values for MCMC samples on the sky
using a clustered-kernel-density estimate (similar to X-Means).

%prep
%setup -q

%build
python setup.py build

%install
python setup.py install --single-version-externally-managed -O1 --root=$RPM_BUILD_ROOT --record=INSTALLED_FILES

%clean
rm -rf $RPM_BUILD_ROOT

%files -f INSTALLED_FILES
%defattr(-,root,root)

%changelog
* Fri Aug 18 2017 Leo Singer <leo.singer@ligo.org> 0.3.2-1

- This release is identical to 0.3.1, except that the RPM and Debian
  changelogs (which are embedded in the source tarball) have been updated.

- Bring back the --seed option and make sure that it works correctly with
  multiprocessing.

- Output sky maps in standard, fixed-resolution HEALPix format rather
  than the more esoteric multi-resolution format.

* Fri Aug 11 2017 Leo Singer <leo.singer@ligo.org> 0.3.0-1

- This release removes several infrequently used command line arguments
  and secondary output data products: 2-step area estimation,
  injection finding, and plotting. This reflects a shift in emphasis
  from calculating accurate areas to calculating accurate sky maps.

- Generate sky maps using a hybrid of the 2D and 3D KDEs. The 2D KDE has
  always produced smaller searched areas than the 3D KDE.

- Restructure the class hierarchy to speed up with Python multiprocessing.

- Add organization, command line, and version to FITS header.

* Tue Oct 18 2016 Leo Singer <leo.singer@ligo.org> 0.2.1-1

- Re-release with packaging distributed in upstream tarball

* Tue Oct 18 2016 Leo Singer <leo.singer@ligo.org> 0.2-1

- ER10 release

* Wed Jun 22 2016 Leo Singer <leo.singer@ligo.org> 0.1-1

- First release
