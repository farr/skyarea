Summary: Compute credible regions on the sky from RA-DEC MCMC samples
Name: skyarea
Version: 0.2.1
Release: 1%{?dist}
Source: https://github.com/farr/skyarea/archive/v%{version}/%{name}-%{version}.tar.gz
License: MIT
Group: Development/Libraries
BuildArch: noarch
Vendor: Will M. Farr <will.farr@ligo.org>
Packager: Leo Singer <leo.singer@ligo.org>
Requires: numpy python-matplotlib scipy healpy glue lalinference-python
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
* Tue Oct 18 2016 Leo Singer <leo.singer@ligo.org> 0.2.1-1

- Re-release with packaging distributed in upstream tarball

* Tue Oct 18 2016 Leo Singer <leo.singer@ligo.org> 0.2-1

- ER10 release

* Wed Jun 22 2016 Leo Singer <leo.singer@ligo.org> 0.1-1

- First release
