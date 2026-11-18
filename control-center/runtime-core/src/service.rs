use crate::manifest::{self, ServiceDef};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Service {
    #[default]
    Sa3,
    Gary,
    Melodyflow,
    StableAudio,
    Carey,
    Foundation,
    Yuey,
}
impl Service {
    pub const ALL: [Self; 7] = [
        Self::Sa3,
        Self::Gary,
        Self::Melodyflow,
        Self::StableAudio,
        Self::Carey,
        Self::Foundation,
        Self::Yuey,
    ];
    pub fn id(self) -> &'static str {
        match self {
            Self::Sa3 => "sa3",
            Self::Gary => "gary",
            Self::Melodyflow => "melodyflow",
            Self::StableAudio => "stable-audio",
            Self::Carey => "carey",
            Self::Foundation => "foundation",
            Self::Yuey => "yuey",
        }
    }
    pub fn parse(id: &str) -> Result<Self, String> {
        Self::ALL
            .into_iter()
            .find(|s| s.id().eq_ignore_ascii_case(id))
            .ok_or_else(|| format!("Unknown service: {id}"))
    }
    pub fn definition(self) -> &'static ServiceDef {
        manifest::bundled()
            .services
            .iter()
            .find(|s| s.id == self.id())
            .expect("bundled service")
    }
    pub fn executable(self) -> &'static str {
        &self
            .definition()
            .native
            .as_ref()
            .expect("native service")
            .executable
    }
    pub fn package(self) -> &'static str {
        match self {
            Self::Sa3 | Self::StableAudio | Self::Foundation => "sa3",
            Self::Gary | Self::Melodyflow => "audiocraft",
            Self::Carey => "acestep",
            Self::Yuey => "yuey",
        }
    }
    pub fn probe_id(self) -> &'static str {
        match self {
            Self::Gary => "musicgen",
            Self::Carey => "acestep",
            Self::Yuey => "yue2",
            Self::StableAudio | Self::Foundation => "sat",
            _ => self.id(),
        }
    }
    pub fn default_port(self) -> u16 {
        if self == Self::Sa3 {
            28300
        } else {
            self.definition().port
        }
    }
    pub fn defaults(self) -> (&'static str, &'static str) {
        match self {
            Self::Sa3 => ("medium", "f16"),
            Self::Gary => ("musicgen-small", "q4_k_m"),
            Self::Melodyflow => ("melodyflow", "f16"),
            Self::StableAudio => ("saos", "q4_k_m"),
            Self::Foundation => ("foundation-1", "q4_k_m"),
            Self::Carey => ("base", "q4_k_m"),
            Self::Yuey => ("yue2", "q4_k_m"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn every_service_has_a_unique_port_and_pinned_default_weights() {
        let mut ports = std::collections::HashSet::new();
        for service in Service::ALL {
            assert!(ports.insert(service.default_port()));
            assert!(!service.executable().is_empty());
            let (model, encoding) = service.defaults();
            let files = if service == Service::Sa3 {
                crate::model_catalog::sa3_medium_files(model, encoding, "f16", "f32")
            } else {
                crate::model_catalog::service_files(service, model, encoding)
            }
            .unwrap();
            assert!(!files.is_empty());
        }
    }
}
