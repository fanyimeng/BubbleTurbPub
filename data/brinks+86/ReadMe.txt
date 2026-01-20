J/A+A/169/14     Hydrogen-line survey of M31. III. HI Holes     (Brinks+, 1986)
================================================================================
A high-resolution hydrogen-line survey of Messier 31.
III. HI holes in the interstellar medium.
    Brinks E., Bajaja E.
   <Astron. Astrophys. 169, 14 (1986)>
   =1986A&A...169...14B    (SIMBAD/NED BibCode)
================================================================================
ADC_Keywords: Galaxies, nearby; H I data; Interstellar medium; Radial velocities
Keywords: Andromeda Galaxy; Astronomical Models; H Lines; Interstellar Gas;
          Galactic Structure; H II Regions; Radio Astronomy; Supernovae;
          Astrophysics

Abstract:
    An analysis is carried out of HI holes in M 31 as detected with the
    Westerbork radiotelescope (WSRT). The holes range from 100-1000pc
    across and represent missing masses ranging from 10^3^ to
    10^7^M_{sun}_, which would require energies ranging from 10 to the
    49th to 10 to the 53rd erg. The kinematic ages are projected to be
    from 2.5-30 million years, with the production rate of the holes being
    1/100000yr. The production rate estimate is used to derive a lower
    limit for the occurrence of supernovae in M 31, about 0.001/yr.
    Finally, a correlation is found between H I holes smaller than 300pc
    and OB associations and H II regions.

Description:
    The work presented here is based on the new high resolution HI survey
    of M31, performed with the Westerbork Synthesis Radio Telescope
    (WSRT). A full account of the observations, the calibrations, and the
    data reduction is given in Paper I (Brinks & Shane 1984A&AS...55..179B).
    The observations were made using the WSRT in its 1.5km configuration.
    Five fields were needed, which were subsequently combined into a
    single map, to cover most of the optical image of M31.

Objects:
    ----------------------------------------------------------
        RA   (ICRS)   DE        Designation(s)
    ----------------------------------------------------------
     00 42 44.33  +41 16 07.5   M31 = NAME Andromeda Galaxy
    ----------------------------------------------------------

File Summary:
--------------------------------------------------------------------------------
 FileName      Lrecl  Records   Explanations
--------------------------------------------------------------------------------
ReadMe            80        .   This file
table2.dat       102      141   Observed properties of HI holes
table3.dat        64      141   Derived properties of HI holes
--------------------------------------------------------------------------------

See also:
 VIII/7  : Hat Creek High-Latitude H I Survey (Heiles+ 1974-1976)
 VII/112 : RC2 Catalogue (de Vaucouleurs+ 1976)
 J/A+AS/31/439 : A Survey of HII Regions in M31 (Pellet+ 1978)
 J/A+AS/61/451 : The 37W catalogue of radio sources in M31 (Walterbos+ 1985)

Byte-by-byte Description of file: table2.dat
--------------------------------------------------------------------------------
  Bytes Format Units    Label    Explanations
--------------------------------------------------------------------------------
  1-  3 I3     ---      Seq      [1/141] Name of the hole (G1)
  5-  9 A5     arcmin   Xpos     X coordinate from the center of the galaxy (1)
 11- 15 A5     arcmin   Ypos     Y coordinate from the center of the galaxy (1)
     17 I1     h        RAh      [0] Center of the hole hour of right
                                   ascension (B1950)
 19- 20 I2     min      RAm      [36/43] Center of the hole minute of right
                                   ascension (B1950)
 22- 25 F4.1   s        RAs      Center of the hole second of right ascension
                                   (B1950)
     27 A1     ---      DE-      [+] Center of the hole sign of declination
                                   (B1950)
 28- 29 I2     deg      DEd      [40/41] Center of the hole degree of
                                   declination (B1950)
 31- 32 I2     arcmin   DEm      Center of the hole arcminute of declination
                                   (B1950)
 34- 35 I2     arcsec   DEs      [0/60] Center of the hole arcsecond of
                                  declination (B1950)
 37- 42 F6.1   km/s     HRV      [-577/342] Radial heliocentric velocity,
                                   V_Hel_ (2)
 44- 48 F5.1   km/s     DV       [-29/25]? Expansion velocity (3)
 50- 53 I4     pc       FWHMI    [80/1000] Full width at half depth along the
                                   minor axis of the hole (4)
 55- 58 I4     pc       FWHMA    [120/1400] Full width at half depth along the
                                   major axis of the hole (4)
 60- 62 I3     deg      psi      [0/170] Position angle, {psi} of the major
                                   axis of the hole, counter-clockwise with
                                   respect to the north declination axis
 64- 67 F4.1   K        INT      [7.5/66] Brightness temperature of the
                                   ambient HI in the channel map at V_Hel_
 69- 72 F4.2   ---      Contr    [0.14/0.74] Contrast of the hole with respect
                                   to its surroundings (5)
     74 I1     ---      Q        [1/3] Quality of the hole (3=high quality)
     76 I1     ---      T        [1/3] Type of the hole (6)
     78 A1     ---      C        Class of the hole ("c"=complete; "p"=partial)
 80-102 A23    ---      Remarks  Remarks
--------------------------------------------------------------------------------
Note (1): Positions of the centre of the hole in standard coordinates X and Y
    with respect to the centre of the galaxy. Most positions are accurate
    to 0.1'.
Note (2): Radial heliocentric velocity at which the hole is most clearly
    defined. This value is accurate to one channel interval or 4.1km/s.
Note (3): Expansion velocity or half the measured velocity splitting in the
    case of a Type 3 hole. In the case of a Type 2 hole the negative sign
    indicates blue-shifted HI. For Type 1 holes no expansion velocity can
    be measured and the entry is left open. The estimated accuracy to
    which DV can be given is about 2km/s.
Note (4): Full width at half depth, FWHMI and FWHMA, measured in parsec along
    the minor and major axes of the hole, not yet corrected for the size
    of the beam. The accuracy is about 20-30pc.
Note (5): The contrast is defined as Contr=(INT-I_50_)/INT, where I_50_ is
    the brightness temperature of the HI at half depth, as determined on
    the basis of the cross-cuts through the channel map.
Note (6): According to the appearance of the holes on the two position-velocity
    maps, three types are defined. See Figures 1-4.
--------------------------------------------------------------------------------

Byte-by-byte Description of file: table3.dat
--------------------------------------------------------------------------------
   Bytes Format Units    Label   Explanations
--------------------------------------------------------------------------------
   1-  3 I3     ---      Seq     [1/141] Name of the hole (G1)
   5-  9 F5.2   kpc      R       [1.7/20.5] Position of the hole in polar
                                  coordinates in the plane of M31
  11- 15 F5.1   deg      theta   [2.6/351.2] Angle measured counter-clockwise
                                  with respect to the north major axis of M31
  17- 20 I4     pc       Maj     [0/1000] Full width at half depth along the
                                  major axis of the hole (7)
  22- 25 I4     pc       Min     [0/1400] Full width at half depth along the
                                  minor axis of the hole (7)
  27- 29 I3     deg      PA      [0/168] Position angle of the hole after beam
                                  deconvolution, measured counter-clockwise with
                                  respect to the north declination axis
  31- 34 I4     pc       Diam    [11/1183]? Effective diameter of the hole (8)
  36- 39 F4.2   ---      Ratio   [0/1]? Axial ratio of the hole (8)
  41- 44 F4.2   ---      nHI     [0.1/1.3] Volume density of the HI surrounding
                                  the hole (9)
  46- 50 F5.1   Myr      Age     [2.2/663.3]? Kinematic age of the hole (10)
  52- 57 F6.1   10+4Msun Mass    [0.1/1165.2]? Indicative HI mass (11)
  59- 64 F6.1   10+43J   Energy  [0.1/1859]? Indicative energy needed to produce
                                  a hole in 10^50^erg (12)
--------------------------------------------------------------------------------
Note (7): Full width at half depth along the minor and minor axis of the hole,
    after deconvolution for the beam of the observing instrument according
    to the formulae given by Wild (1970AuJPh..23..113W).
Note (8): Effective diameter of the hole defined as the geometric mean of the
    dimensions of the hole listed in columns Maj and Min, i.e.
    Diam=(MinxMaj)^1/2^, and the axial ratio of the hole defined as
    Ratio=Min/Maj. In the case of an unresolved hole the entry for Diam is
    left open, and the ratio is 0.0.
Note (9): Volume density of the HI surrounding the hole estimated on the basis
    of the intensity in the integrated HI map and transformed to
    atoms/cm^3^ according to formula (1):

    n_HI_=1.823x10^18^I_B_cos(i/h)(2{pi})^0.5^

    where I_B_ is the total surface brightness in K.km/s and the factor 
    cos(i/h)(2{pi})^0.5^ converts the measured column density to volume density.
Note (10): Kinematic age of the hole in units of 10^6^yr defined as
    Age=Diam/(2xDV). This age estimate represents an upper limit to the
    age of a hole. If the evolution of a hole follows that of an expanding
    shell of swept-up interstellar material in the snowplough phase as
    described by Weaver+ (1977ApJ...218..377W; see also
    Bruhweiler+ 1980ApJ...238L..27B), the age as listed in the table
    should be reduced to 60% of its value.
Note (11): We evaluate the mass which might have been present at the position
    of the hole under the assumption that the hole is entirely empty, that
    {pi}/6(Diam)^3^ is the characteristic volume of the hole, and that
    n_HI_ is the average HI volume density if the hole were not present.
Note (12): To facilitate comparisons with the work by
    Heiles (1979ApJ...229..533H) for the shells in our Galaxy we use the
    same expression as used by him for the energy deposited in the
    interstellar medium, i.e.

    E=5.3x10^43^n_HI_^1.12^(Diam/2)^3.12^DV^1.4^erg
--------------------------------------------------------------------------------

Global notes:
Note (G1): The holes are numbered according to their position in Xpos, starting
    from positions south of the center of the galaxy.
--------------------------------------------------------------------------------

Nomenclature Notes: Table 2, <[BB86] NNN> in Simbad.

History:
    Tables scanned by CDS.

References:
    Brinks & Shane    Paper I.     1984A&AS...55..179B
    Brinks & Burton   Paper II.    1984A&A...141..195B
    Brinks & Bajaja   Paper III.   1986A&A...169...14B   This catalog

================================================================================
(End)                                     Emmanuelle Perret [CDS]    05-Mar-2025
